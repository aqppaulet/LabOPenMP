#include "odd_even.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void comparar_intercambiar(int *a, size_t i)
{
    if (a[i] > a[i + 1]) {
        const int aux = a[i];
        a[i] = a[i + 1];
        a[i + 1] = aux;
    }
}

void odd_even_secuencial(int *a, size_t n)
{
    if (n < 2)
        return;

    for (size_t fase = 0; fase < n; ++fase) {
        const size_t inicio = fase % 2;
        for (size_t i = inicio; i < n - 1; i += 2)
            comparar_intercambiar(a, i);
    }
}

void odd_even_paralelo(int *a, size_t n, int threads,
                      enum odd_even_plan plan)
{
    if (n < 2)
        return;
    if (threads < 1)
        threads = 1;

    /* Todos los threads ejecutan las mismas fases. Dentro de una fase los
     * pares son disjuntos; la barrera de omp for impide solapar dos fases.
     */
    #pragma omp parallel num_threads(threads) default(none) shared(a, n, plan)
    {
        for (size_t fase = 0; fase < n; ++fase) {
            const size_t inicio = fase % 2;
            if (plan == ODD_EVEN_DYNAMIC) {
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = inicio; i < n - 1; i += 2)
                    comparar_intercambiar(a, i);
            } else {
                #pragma omp for schedule(static)
                for (size_t i = inicio; i < n - 1; i += 2)
                    comparar_intercambiar(a, i);
            }
        }
    }
}

static int comparar_enteros(const void *izquierda, const void *derecha)
{
    const int a = *(const int *)izquierda;
    const int b = *(const int *)derecha;
    /* Evita el desbordamiento que podria causar devolver a-b. */
    return (a > b) - (a < b);
}

int odd_even_verificar(const int *entrada, const int *salida, size_t n)
{
    if (n == 0)
        return 1;
    if (entrada == NULL || salida == NULL)
        return 0;
    if (n > SIZE_MAX / sizeof(int))
        return -1;

    int *referencia = malloc(n * sizeof(*referencia));
    if (referencia == NULL)
        return -1;
    memcpy(referencia, entrada, n * sizeof(*referencia));
    qsort(referencia, n, sizeof(*referencia), comparar_enteros);

    int correcto = 1;
    for (size_t i = 0; i < n; ++i) {
        if (referencia[i] != salida[i]) {
            correcto = 0;
            break;
        }
    }
    free(referencia);
    return correcto;
}

static int probar_caso(FILE *salida, const char *nombre,
                      const int *entrada, size_t n)
{
    int *copia = NULL;
    if (n > 0) {
        copia = malloc(n * sizeof(*copia));
        if (copia == NULL) {
            fprintf(salida, "[FALLO] odd-even: sin memoria en %s.\n", nombre);
            return 0;
        }
        memcpy(copia, entrada, n * sizeof(*copia));
    }
    odd_even_secuencial(copia, n);
    if (odd_even_verificar(entrada, copia, n) != 1) {
        fprintf(salida, "[FALLO] odd-even secuencial: %s.\n", nombre);
        free(copia);
        return 0;
    }

    const int cantidades[] = {1, 2, 4, 8};
    for (size_t p = 0; p < sizeof(cantidades) / sizeof(cantidades[0]); ++p) {
        for (int politica = 0; politica < 2; ++politica) {
            if (n > 0)
                memcpy(copia, entrada, n * sizeof(*copia));
            odd_even_paralelo(copia, n, cantidades[p],
                              politica == 0 ? ODD_EVEN_STATIC : ODD_EVEN_DYNAMIC);
            if (odd_even_verificar(entrada, copia, n) != 1) {
                fprintf(salida,
                        "[FALLO] odd-even: %s, threads=%d, politica=%s.\n",
                        nombre, cantidades[p], politica == 0 ? "static" : "dynamic,64");
                free(copia);
                return 0;
            }
        }
    }
    fprintf(salida, "[OK] odd-even: %-20s n=%zu; secuencial, static y dynamic,64; h=1,2,4,8.\n",
            nombre, n);
    free(copia);
    return 1;
}

static uint32_t siguiente_aleatorio(uint32_t *estado)
{
    uint32_t x = *estado;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *estado = x;
    return x;
}

int odd_even_autoprueba(FILE *salida)
{
    if (salida == NULL)
        salida = stdout;

    static const int uno[] = {42};
    static const int dos_ordenados[] = {-1, 2};
    static const int dos_inversos[] = {2, -1};
    static const int inverso_par[] = {8, 7, 6, 5, 4, 3, 2, 1};
    static const int inverso_impar[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    static const int duplicados[] = {3, -1, 3, 0, -1, 7, 7, 3, 0};
    static const int iguales[] = {5, 5, 5, 5, 5, 5};
    static const int extremos[] = {INT_MAX, 0, INT_MIN, -1, INT_MAX, INT_MIN};
    static const int ordenado[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    struct caso {
        const char *nombre;
        const int *entrada;
        size_t n;
    };
    const struct caso casos[] = {
        {"vacio", NULL, 0},
        {"un elemento", uno, 1},
        {"dos ordenados", dos_ordenados, 2},
        {"dos inversos", dos_inversos, 2},
        {"inverso par", inverso_par, 8},
        {"inverso impar", inverso_impar, 9},
        {"duplicados", duplicados, 9},
        {"todos iguales", iguales, 6},
        {"enteros extremos", extremos, 6},
        {"ya ordenado", ordenado, 10}
    };
    for (size_t c = 0; c < sizeof(casos) / sizeof(casos[0]); ++c) {
        if (!probar_caso(salida, casos[c].nombre, casos[c].entrada, casos[c].n))
            return 0;
    }

    uint32_t estado = UINT32_C(2026);
    int aleatorios[257];
    for (size_t i = 0; i < sizeof(aleatorios) / sizeof(aleatorios[0]); ++i)
        aleatorios[i] = (int)(siguiente_aleatorio(&estado) % 201) - 100;
    if (!probar_caso(salida, "aleatorio par", aleatorios, 256)
        || !probar_caso(salida, "aleatorio impar", aleatorios, 257))
        return 0;

    /* Verifica tambien que el comprobador rechaza una salida ordenada que
     * pierde elementos: comprobar solo orden o suma no seria suficiente.
     */
    static const int entrada_control[] = {1, 2, 3};
    static const int salida_incorrecta[] = {1, 1, 4};
    if (odd_even_verificar(entrada_control, salida_incorrecta, 3) != 0) {
        fprintf(salida, "[FALLO] odd-even: verificador de multiconjunto.\n");
        return 0;
    }
    fprintf(salida, "[OK] odd-even: el verificador detecta elementos alterados.\n");
    return 1;
}
