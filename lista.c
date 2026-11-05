#include "lista.h"

#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int key;
    struct Node *next;
    omp_lock_t lock;
} Node;

typedef struct {
    int key;
    int deletion;
} Operation;

struct ListWorkload {
    size_t nodes;
    size_t count;
    size_t searches;
    size_t hits;
    size_t deletions;
    Operation *operations;
    unsigned char *removed;
};

static uint32_t random_next(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void list_free(Node **head)
{
    while (*head != NULL) {
        Node *old = *head;
        *head = old->next;
        omp_destroy_lock(&old->lock);
        free(old);
    }
}

/* Operaciones secuenciales de una lista ordenada sin duplicados. */
static int list_insert(Node **head, int key)
{
    Node **link = head;
    while (*link != NULL && (*link)->key < key)
        link = &(*link)->next;
    if (*link != NULL && (*link)->key == key)
        return 0;
    Node *node = malloc(sizeof(*node));
    if (node == NULL)
        return -1;
    node->key = key;
    node->next = *link;
    omp_init_lock(&node->lock);
    *link = node;
    return 1;
}

static int list_member(const Node *head, int key)
{
    while (head != NULL && head->key < key)
        head = head->next;
    return head != NULL && head->key == key;
}

static int list_delete(Node **head, int key)
{
    Node **link = head;
    while (*link != NULL && (*link)->key < key)
        link = &(*link)->next;
    if (*link == NULL || (*link)->key != key)
        return 0;
    Node *old = *link;
    *link = old->next;
    omp_destroy_lock(&old->lock);
    free(old);
    return 1;
}

static int list_initial(Node **head, size_t count)
{
    Node **tail = head;
    *head = NULL;
    for (size_t i = 0; i < count; ++i) {
        Node *node = malloc(sizeof(*node));
        if (node == NULL) {
            list_free(head);
            return 0;
        }
        node->key = (int)i;
        node->next = NULL;
        omp_init_lock(&node->lock);
        *tail = node;
        tail = &node->next;
    }
    return 1;
}

ListWorkload *list_workload_create(size_t nodes, size_t operations,
                                 unsigned search_percent, unsigned seed)
{
    if (nodes < 2 || nodes > (size_t)INT_MAX / 2 || search_percent > 100 ||
        operations > SIZE_MAX / sizeof(Operation))
        return NULL;
    size_t deletions = (operations / 100) * (100 - search_percent) +
                       ((operations % 100) * (100 - search_percent)) / 100;
    const size_t even_count = (nodes + 1) / 2;
    if (deletions > even_count)
        return NULL;
    ListWorkload *work = calloc(1, sizeof(*work));
    if (work == NULL)
        return NULL;
    work->nodes = nodes;
    work->count = operations;
    work->searches = operations - deletions;
    work->deletions = deletions;
    work->operations = calloc(operations != 0 ? operations : 1,
                              sizeof(*work->operations));
    work->removed = calloc(nodes, sizeof(*work->removed));
    int *pool = malloc(even_count * sizeof(*pool));
    if (work->operations == NULL || work->removed == NULL || pool == NULL) {
        free(pool);
        list_workload_destroy(work);
        return NULL;
    }
    uint32_t state = seed != 0 ? seed : UINT32_C(0x9e3779b9);
    for (size_t i = 0; i < even_count; ++i)
        pool[i] = (int)(2 * i);
    for (size_t i = even_count; i > 1; --i) {
        size_t j = random_next(&state) % i;
        int temporary = pool[i - 1];
        pool[i - 1] = pool[j];
        pool[j] = temporary;
    }
    for (size_t i = 0; i < deletions; ++i) {
        work->operations[i].deletion = 1;
        work->operations[i].key = pool[i];
        work->removed[pool[i]] = 1;
    }
    free(pool);
    for (size_t i = deletions; i < operations; ++i) {
        /* Impares presentes y claves fuera de la lista. Las búsquedas no
         * dependen de si una eliminación ya se ejecutó o se ejecutará. */
        if (random_next(&state) & 1U) {
            work->operations[i].key =
                (int)(2 * (random_next(&state) % (nodes / 2)) + 1);
            ++work->hits;
        } else {
            work->operations[i].key =
                (int)(nodes + random_next(&state) % nodes);
        }
    }
    for (size_t i = operations; i > 1; --i) {
        size_t j = random_next(&state) % i;
        Operation temporary = work->operations[i - 1];
        work->operations[i - 1] = work->operations[j];
        work->operations[j] = temporary;
    }
    return work;
}

void list_workload_destroy(ListWorkload *workload)
{
    if (workload != NULL) {
        free(workload->operations);
        free(workload->removed);
        free(workload);
    }
}

/* Primero se descartan ciclos; después se comprueba el conjunto exacto.
 * La igualdad de la secuencia ordenada también prueba presentes y ausentes. */
static int list_verify(const Node *head, const ListWorkload *work, size_t *count)
{
    const Node *slow = head;
    const Node *fast = head;
    while (fast != NULL && fast->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast)
            return 0;
    }
    *count = 0;
    const Node *node = head;
    for (size_t key = 0; key < work->nodes; ++key) {
        if (work->removed[key])
            continue;
        if (node == NULL || node->key != (int)key)
            return 0;
        ++*count;
        node = node->next;
    }
    return node == NULL && *count == work->nodes - work->deletions;
}

/* Acoplamiento de bloqueos: se adquiere el nodo siguiente antes de soltar
 * el actual. head_guard protege el enlace inicial, sin valor centinela. */
static int list_member_nodes(Node **head, omp_lock_t *head_guard, int key)
{
    omp_set_lock(head_guard);
    Node *current = *head;
    if (current != NULL)
        omp_set_lock(&current->lock);
    omp_unset_lock(head_guard);
    while (current != NULL && current->key < key) {
        Node *next = current->next;
        if (next != NULL)
            omp_set_lock(&next->lock);
        omp_unset_lock(&current->lock);
        current = next;
    }
    int found = current != NULL && current->key == key;
    if (current != NULL)
        omp_unset_lock(&current->lock);
    return found;
}

static int list_delete_nodes(Node **head, omp_lock_t *head_guard, int key)
{
    Node *previous = NULL;
    omp_lock_t *previous_guard = head_guard;
    omp_set_lock(previous_guard);
    Node *current = *head;
    if (current != NULL)
        omp_set_lock(&current->lock);

    while (current != NULL && current->key < key) {
        Node *next = current->next;
        if (next != NULL)
            omp_set_lock(&next->lock);
        omp_unset_lock(previous_guard);
        previous = current;
        previous_guard = &previous->lock;
        current = next;
    }
    int removed = current != NULL && current->key == key;
    if (removed) {
        if (previous != NULL)
            previous->next = current->next;
        else
            *head = current->next;
        /* Con predecesor y actual bloqueados no hay otros usuarios ni
         * esperadores del actual: llegar a él exige bloquear predecesor.
         * Se destruye y libera ANTES de soltar esa guardia. */
        omp_unset_lock(&current->lock);
        omp_destroy_lock(&current->lock);
        free(current);
    } else if (current != NULL) {
        omp_unset_lock(&current->lock);
    }
    omp_unset_lock(previous_guard);
    return removed;
}

int list_run(const ListWorkload *work, ListStrategy strategy,
             int threads, ListResult *result)
{
    if (work == NULL || result == NULL || threads < 1 ||
        (strategy != LIST_SEQ && strategy != LIST_GLOBAL &&
         strategy != LIST_NODE_LOCKS))
        return 0;
    memset(result, 0, sizeof(*result));
    Node *head;
    if (!list_initial(&head, work->nodes))
        return 0;

    size_t searches = 0, hits = 0, deletions = 0, removed = 0;
    int threads_used = 1;
    omp_lock_t lock;
    omp_lock_t head_guard;
    if (strategy == LIST_GLOBAL)
        omp_init_lock(&lock);
    if (strategy == LIST_NODE_LOCKS)
        omp_init_lock(&head_guard);

    const double start = omp_get_wtime();
    if (strategy == LIST_SEQ) {
        for (size_t i = 0; i < work->count; ++i) {
            Operation operation = work->operations[i];
            if (operation.deletion) {
                ++deletions;
                removed += (size_t)list_delete(&head, operation.key);
            } else {
                ++searches;
                hits += (size_t)list_member(head, operation.key);
            }
        }
    } else if (strategy == LIST_GLOBAL) {
        #pragma omp parallel num_threads(threads) default(none) \
            shared(head, work, lock, threads_used) \
            reduction(+:searches, hits, deletions, removed)
        {
            #pragma omp single
            threads_used = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < work->count; ++i) {
                Operation operation = work->operations[i];
                /* También se protege cada búsqueda completa: un escritor
                 * no puede liberar ningún nodo mientras se recorre. */
                omp_set_lock(&lock);
                int success = operation.deletion ?
                    list_delete(&head, operation.key) :
                    list_member(head, operation.key);
                omp_unset_lock(&lock);
                if (operation.deletion) {
                    ++deletions;
                    removed += (size_t)success;
                } else {
                    ++searches;
                    hits += (size_t)success;
                }
            }
        }
    } else {
        #pragma omp parallel num_threads(threads) default(none) \
            shared(head, work, head_guard, threads_used) \
            reduction(+:searches, hits, deletions, removed)
        {
            #pragma omp single
            threads_used = omp_get_num_threads();
            #pragma omp for schedule(static)
            for (size_t i = 0; i < work->count; ++i) {
                Operation operation = work->operations[i];
                int success;
                if (operation.deletion) {
                    success = list_delete_nodes(&head, &head_guard, operation.key);
                    ++deletions;
                    removed += (size_t)success;
                } else {
                    success = list_member_nodes(&head, &head_guard, operation.key);
                    ++searches;
                    hits += (size_t)success;
                }
            }
        }
    }
    result->seconds = omp_get_wtime() - start;
    if (strategy == LIST_GLOBAL)
        omp_destroy_lock(&lock);
    if (strategy == LIST_NODE_LOCKS)
        omp_destroy_lock(&head_guard);

    result->searches = searches;
    result->hits = hits;
    result->deletions = deletions;
    result->removed = removed;
    result->threads_used = threads_used;
    result->valid = list_verify(head, work, &result->final_nodes) &&
        searches == work->searches && hits == work->hits &&
        deletions == work->deletions && removed == work->deletions;
    list_free(&head);
    return 1;
}

/* Prueba separada del benchmark: Member(k) y Delete(k) usan la misma clave.
 * Un bloqueo externo POR CLAVE protege el oráculo; no serializa claves
 * distintas, cuyos recorridos y eliminaciones continúan siendo concurrentes.
 * Estos bloqueos adicionales se usan SOLO aquí, nunca al medir rendimiento. */
static int list_overlap_test(ListStrategy strategy, int threads, unsigned seed)
{
    ListWorkload *work = list_workload_create(257, 1024, 88, seed);
    if (work == NULL)
        return 0;
    uint32_t state = seed;
    for (size_t i = 0; i < work->count; ++i) {
        if (!work->operations[i].deletion)
            work->operations[i].key = (int)(random_next(&state) % 321);
    }
    Node *head;
    if (!list_initial(&head, work->nodes)) {
        list_workload_destroy(work);
        return 0;
    }
    unsigned char present[257];
    memset(present, 1, sizeof(present));
    size_t errors = 0;
    omp_lock_t lock;
    omp_lock_t head_guard;
    omp_lock_t key_locks[321];
    for (size_t key = 0; key < 321; ++key)
        omp_init_lock(&key_locks[key]);
    if (strategy == LIST_GLOBAL)
        omp_init_lock(&lock);
    else
        omp_init_lock(&head_guard);

    #pragma omp parallel for num_threads(threads) schedule(static) default(none) \
        shared(head, work, present, strategy, lock, head_guard, key_locks) \
        reduction(+:errors)
    for (size_t i = 0; i < work->count; ++i) {
        Operation operation = work->operations[i];
        omp_set_lock(&key_locks[operation.key]);
        if (strategy == LIST_GLOBAL) {
            omp_set_lock(&lock);
        }

        int expected = (size_t)operation.key < work->nodes ?
            present[operation.key] : 0;
        int actual;
        if (strategy == LIST_GLOBAL) {
            actual = operation.deletion ?
                list_delete(&head, operation.key) :
                list_member(head, operation.key);
        } else {
            actual = operation.deletion ?
                list_delete_nodes(&head, &head_guard, operation.key) :
                list_member_nodes(&head, &head_guard, operation.key);
        }
        if (actual != expected)
            ++errors;
        if (operation.deletion)
            present[operation.key] = 0;

        if (strategy == LIST_GLOBAL)
            omp_unset_lock(&lock);
        omp_unset_lock(&key_locks[operation.key]);
    }
    if (strategy == LIST_GLOBAL)
        omp_destroy_lock(&lock);
    else
        omp_destroy_lock(&head_guard);
    for (size_t key = 0; key < 321; ++key)
        omp_destroy_lock(&key_locks[key]);

    size_t final_nodes;
    int valid = errors == 0 && list_verify(head, work, &final_nodes);
    for (size_t key = 0; key < work->nodes; ++key)
        valid &= present[key] == !work->removed[key];
    list_free(&head);
    list_workload_destroy(work);
    return valid;
}

static int list_node_edge_test(void)
{
    Node *head = NULL;
    omp_lock_t head_guard;
    omp_init_lock(&head_guard);
    int passed = list_insert(&head, INT_MIN) == 1;
    passed &= list_insert(&head, 0) == 1;
    passed &= list_insert(&head, INT_MAX) == 1;
    passed &= list_member_nodes(&head, &head_guard, INT_MIN);
    passed &= list_member_nodes(&head, &head_guard, INT_MAX);
    passed &= !list_delete_nodes(&head, &head_guard, -1);
    passed &= list_delete_nodes(&head, &head_guard, INT_MIN);
    passed &= list_delete_nodes(&head, &head_guard, INT_MAX);
    passed &= list_delete_nodes(&head, &head_guard, 0);
    passed &= head == NULL;
    passed &= !list_member_nodes(&head, &head_guard, 0);
    passed &= !list_delete_nodes(&head, &head_guard, 0);
    list_free(&head);
    omp_destroy_lock(&head_guard);
    return passed;
}

int list_selftest(FILE *output)
{
    int passed = 1;
    Node *head = NULL;
    passed &= list_insert(&head, 4) == 1;
    passed &= list_insert(&head, 2) == 1;
    passed &= list_insert(&head, 6) == 1;
    passed &= list_insert(&head, 2) == 0;
    passed &= list_member(head, 2) && list_member(head, 4) &&
              list_member(head, 6) && !list_member(head, 3);
    passed &= list_delete(&head, 4) && list_delete(&head, 2) &&
              list_delete(&head, 6) && !list_delete(&head, 6) && head == NULL;
    list_free(&head);
    passed &= list_node_edge_test();
    if (output != NULL)
        fprintf(output, "Lista: operaciones basicas, enteros extremos y vaciado por nodo: %s\n",
                passed ? "OK" : "FALLO");

    const unsigned ratios[] = {100, 95, 80, 50};
    const int thread_counts[] = {1, 2, 4, 8};
    size_t cases = 0;
    for (size_t ratio = 0; ratio < sizeof(ratios) / sizeof(ratios[0]); ++ratio) {
        /* 257 nodos también comprueba el último nodo par de longitud impar. */
        ListWorkload *work = list_workload_create(257, 256, ratios[ratio], 2026);
        if (work == NULL) {
            passed = 0;
            continue;
        }
        for (int strategy = LIST_SEQ; strategy <= LIST_NODE_LOCKS; ++strategy) {
            size_t limit = strategy == LIST_SEQ ? 1 :
                sizeof(thread_counts) / sizeof(thread_counts[0]);
            for (size_t t = 0; t < limit; ++t) {
                ListResult result;
                passed &= list_run(work, (ListStrategy)strategy,
                                   thread_counts[t], &result) && result.valid;
                ++cases;
            }
        }
        list_workload_destroy(work);
    }
    /* Cero operaciones y lista mínima. */
    ListWorkload *empty = list_workload_create(2, 0, 100, 0);
    if (empty == NULL) {
        passed = 0;
    } else {
        ListResult result;
        passed &= list_run(empty, LIST_NODE_LOCKS, 4, &result) && result.valid;
        list_workload_destroy(empty);
        ++cases;
    }
    if (output != NULL)
        fprintf(output, "Lista: %zu casos, conjunto final exacto, contadores y ausencia de ciclos: %s\n",
                cases, passed ? "OK" : "FALLO");

    int overlap_passed = 1;
    size_t overlap_cases = 0;
    for (int strategy = LIST_GLOBAL; strategy <= LIST_NODE_LOCKS; ++strategy) {
        for (size_t t = 1; t < sizeof(thread_counts) / sizeof(thread_counts[0]); ++t) {
            for (unsigned repetition = 0; repetition < 3; ++repetition) {
                overlap_passed &= list_overlap_test((ListStrategy)strategy,
                    thread_counts[t], 2026 + repetition);
                ++overlap_cases;
            }
        }
    }
    if (output != NULL)
        fprintf(output, "Lista: %zu pruebas Member/Delete con mismas claves y oraculo protegido: %s\n",
                overlap_cases, overlap_passed ? "OK" : "FALLO");
    passed &= overlap_passed;
    return passed;
}
