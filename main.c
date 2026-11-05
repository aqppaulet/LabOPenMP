#include "lista.h"
#include "odd_even.h"

#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_REPS 30
#define THREAD_COUNTS 4
static const int thread_counts[THREAD_COUNTS] = {1, 2, 4, 8};

typedef struct {
    const char *problem, *method;
    size_t n, operations;
    unsigned searches;
    int threads, used;
    double times[MAX_REPS];
} Measurement;

static FILE *raw_file;
static FILE *summary_file;

static double average(const Measurement *m, int reps) {
    double total = 0.0;
    for (int r = 0; r < reps; ++r) total += m->times[r];
    return total / reps;
}

static void write_raw(const Measurement *m, int rep, int valid) {
    fprintf(raw_file, "%s,%s,%zu,%zu,%u,%d,%d,%d,%.9f,%d\n",
            m->problem, m->method, m->n, m->operations, m->searches,
            m->threads, m->used, rep + 1, m->times[rep], valid);
    fflush(raw_file);
}

static void write_summary(const Measurement *m, int reps, double seq) {
    double mean = average(m, reps), min = m->times[0], max = min;
    for (int r = 1; r < reps; ++r) {
        if (m->times[r] < min) min = m->times[r];
        if (m->times[r] > max) max = m->times[r];
    }
    double speedup = mean > 0.0 ? seq / mean : 0.0;
    fprintf(summary_file,
            "%s,%s,%zu,%zu,%u,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f\n",
            m->problem, m->method, m->n, m->operations, m->searches,
            m->threads, m->used, reps, seq, mean, min, max,
            speedup, speedup / m->used);
    fflush(summary_file);
}

static int bench_list(int reps) {
    const size_t nodes = 12000, operations = 24000;
    const unsigned percentages[] = {95, 80};
    for (size_t scenario = 0; scenario < 2; ++scenario) {
        unsigned percent = percentages[scenario];
        ListWorkload *w = list_workload_create(nodes, operations, percent, 2026);
        if (!w) return 0;
        Measurement m[1 + 2 * THREAD_COUNTS] = {0};
        for (int k = 0; k < 1 + 2 * THREAD_COUNTS; ++k) {
            m[k].problem = "lista";
            m[k].method = k == 0 ? "secuencial" : k <= THREAD_COUNTS ? "global" : "rwlock";
            m[k].n = nodes;
            m[k].operations = operations;
            m[k].searches = percent;
            m[k].threads = k == 0 ? 1 : thread_counts[(k - 1) % THREAD_COUNTS];
        }
        ListResult warmup;
        if (!list_run(w, LIST_SEQ, 1, &warmup) || !warmup.valid) {
            list_workload_destroy(w);
            return 0;
        }
        for (int rep = 0; rep < reps; ++rep) {
            /* Rotar el orden reduce el sesgo de ejecutar siempre primero la base. */
            for (int offset = 0; offset < 1 + 2 * THREAD_COUNTS; ++offset) {
                int k = (offset + rep) % (1 + 2 * THREAD_COUNTS);
                ListStrategy strategy = k == 0 ? LIST_SEQ : k <= THREAD_COUNTS ? LIST_GLOBAL : LIST_RWLOCK;
                ListResult result;
                if (!list_run(w, strategy, m[k].threads, &result) || !result.valid) {
                    fprintf(stderr, "Fallo de correctitud en lista, metodo %s.\n", m[k].method);
                    list_workload_destroy(w);
                    return 0;
                }
                m[k].used = result.threads_used;
                if (m[k].used != m[k].threads) {
                    fprintf(stderr, "Se solicitaron %d hilos y se obtuvieron %d.\n", m[k].threads, m[k].used);
                    list_workload_destroy(w);
                    return 0;
                }
                m[k].times[rep] = result.seconds;
                write_raw(&m[k], rep, result.valid);
            }
        }
        double seq = average(&m[0], reps);
        for (int k = 0; k < 1 + 2 * THREAD_COUNTS; ++k) write_summary(&m[k], reps, seq);
        printf("Lista: %u%% busquedas, %u%% eliminaciones, %zu nodos: OK (%d repeticiones).\n",
               percent, 100 - percent, nodes, reps);
        fflush(stdout);
        list_workload_destroy(w);
    }
    return 1;
}

static uint32_t next_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static int bench_sort(int reps) {
    const size_t sizes[] = {1000, 4000, 12000};
    for (size_t scenario = 0; scenario < 3; ++scenario) {
        size_t n = sizes[scenario];
        int *input = malloc(n * sizeof(*input)), *a = malloc(n * sizeof(*a));
        if (!input || !a) { free(input); free(a); return 0; }
        uint32_t seed = 2026;
        for (size_t i = 0; i < n; ++i) input[i] = (int)(next_random(&seed) % 20001) - 10000;
        Measurement m[1 + 2 * THREAD_COUNTS] = {0};
        for (int k = 0; k < 1 + 2 * THREAD_COUNTS; ++k) {
            m[k].problem = "odd_even";
            m[k].method = k == 0 ? "secuencial" : k <= THREAD_COUNTS ? "static" : "dynamic64";
            m[k].n = n;
            m[k].threads = k == 0 ? 1 : thread_counts[(k - 1) % THREAD_COUNTS];
            m[k].used = m[k].threads;
        }
        memcpy(a, input, n * sizeof(*a));
        odd_even_secuencial(a, n); /* Calentamiento sin registrar. */
        for (int rep = 0; rep < reps; ++rep) {
            for (int offset = 0; offset < 1 + 2 * THREAD_COUNTS; ++offset) {
                int k = (offset + rep) % (1 + 2 * THREAD_COUNTS);
                memcpy(a, input, n * sizeof(*a));
                double start = omp_get_wtime();
                if (k == 0) odd_even_secuencial(a, n);
                else odd_even_paralelo(a, n, m[k].threads,
                    k <= THREAD_COUNTS ? ODD_EVEN_STATIC : ODD_EVEN_DYNAMIC);
                m[k].times[rep] = omp_get_wtime() - start;
                int valid = odd_even_verificar(input, a, n);
                write_raw(&m[k], rep, valid == 1);
                if (valid != 1) {
                    fprintf(stderr, "Fallo de correctitud en Odd-Even.\n");
                    free(input); free(a); return 0;
                }
            }
        }
        double seq = average(&m[0], reps);
        for (int k = 0; k < 1 + 2 * THREAD_COUNTS; ++k) write_summary(&m[k], reps, seq);
        printf("Odd-Even: n=%zu: orden y elementos OK (%d repeticiones).\n", n, reps);
        fflush(stdout);
        free(input); free(a);
    }
    return 1;
}

static int check_teams(void) {
    for (int i = 0; i < THREAD_COUNTS; ++i) {
        int requested = thread_counts[i], used = 0;
        #pragma omp parallel num_threads(requested) shared(used)
        {
            #pragma omp single
            used = omp_get_num_threads();
        }
        if (used != requested) {
            fprintf(stderr, "OpenMP no pudo crear %d hilos (creo %d). Revise OMP_THREAD_LIMIT.\n", requested, used);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        int a = list_selftest(stdout);
        int b = odd_even_autoprueba(stdout);
        return a && b ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if ((argc == 2 || argc == 3) && strcmp(argv[1], "--bench") == 0) {
        int reps = 5;
        if (argc == 3) {
            char *end;
            errno = 0;
            long parsed = strtol(argv[2], &end, 10);
            if (errno || *end != '\0' || end == argv[2] || parsed < 1 || parsed > MAX_REPS) {
                fprintf(stderr, "Repeticiones: entero de 1 a %d.\n", MAX_REPS);
                return EXIT_FAILURE;
            }
            reps = (int)parsed;
        }
        if (!check_teams()) return EXIT_FAILURE;
        if (mkdir("resultados", 0775) != 0 && errno != EEXIST) { perror("resultados"); return EXIT_FAILURE; }
        raw_file = fopen("resultados/mediciones.csv", "w");
        summary_file = fopen("resultados/resumen.csv", "w");
        if (!raw_file || !summary_file) {
            perror("CSV");
            if (raw_file) fclose(raw_file);
            if (summary_file) fclose(summary_file);
            return EXIT_FAILURE;
        }
        fputs("problema,metodo,n,operaciones,busquedas_pct,hilos,hilos_reales,repeticion,segundos,correcto\n", raw_file);
        fputs("problema,metodo,n,operaciones,busquedas_pct,hilos,hilos_reales,repeticiones,tseq_s,tmedio_s,tmin_s,tmax_s,speedup,eficiencia\n", summary_file);
        int ok = bench_list(reps) && bench_sort(reps);
        if (fclose(raw_file) != 0) ok = 0;
        if (fclose(summary_file) != 0) ok = 0;
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    printf("Uso: %s --test | --bench [repeticiones: 1..%d; defecto 5]\n", argv[0], MAX_REPS);
    return argc == 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}
