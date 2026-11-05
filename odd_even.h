#ifndef ODD_EVEN_H
#define ODD_EVEN_H

#include <stddef.h>
#include <stdio.h>

enum odd_even_plan {
    ODD_EVEN_STATIC,
    ODD_EVEN_DYNAMIC
};

/* Se ejecutan n fases alternadas, empezando por (0,1), (2,3), ... . */
void odd_even_secuencial(int *a, size_t n);

/* Una region paralela; una barrera implicita al terminar cada fase.
 * DYNAMIC usa bloques de 64 comparaciones. threads <= 0 se trata como 1.
 * Para n < 2, a puede ser NULL; para n >= 2 debe apuntar a n enteros.
 */
void odd_even_paralelo(int *a, size_t n, int threads,
                      enum odd_even_plan plan);

/* Contrasta con qsort: comprueba orden y conservacion exacta del multiconjunto.
 * Retorna 1 si coincide, 0 si no coincide y -1 si no se puede reservar memoria.
 */
int odd_even_verificar(const int *entrada, const int *salida, size_t n);

/* Casos pequenos y aleatorios, ambas politicas y 1, 2, 4, 8 threads.
 * Retorna 1 si todas las pruebas pasan; 0 en caso contrario.
 */
int odd_even_autoprueba(FILE *salida);

#endif
