#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "Agent.h"


typedef struct {
  Agent *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
    
  a->array = calloc(initialSize, sizeof(Agent));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Array *a, Agent element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = realloc(a->array, a->size * sizeof(Agent));
  }
  a->array[a->used++] = element;
}

void freeArray(Array *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}