#ifndef LISTA_H
#define LISTA_H

#include <stddef.h>
#include <stdio.h>

typedef enum { LIST_SEQ, LIST_GLOBAL, LIST_NODE_LOCKS } ListStrategy;
typedef struct ListWorkload ListWorkload;

typedef struct {
    double seconds;
    size_t searches;
    size_t hits;
    size_t deletions;
    size_t removed;
    size_t final_nodes;
    int valid;
    int threads_used;
} ListResult;

/* La lista inicial contiene 0,...,nodes-1. Las eliminaciones usan pares
 * distintos; las búsquedas usan impares presentes o claves ausentes.
 * Se exige nodes >= 2 y como máximo ceil(nodes/2) eliminaciones. */
ListWorkload *list_workload_create(size_t nodes, size_t operations,
                                 unsigned search_percent, unsigned seed);
void list_workload_destroy(ListWorkload *workload);

/* Construcción y verificación no se incluyen en seconds.
 * Devuelve 1 si se ejecutó, 0 ante parámetros inválidos o falta de memoria.
 * valid informa si los contadores y TODOS los nodos finales son correctos. */
int list_run(const ListWorkload *workload, ListStrategy strategy,
             int threads, ListResult *result);
int list_selftest(FILE *output);

#endif
