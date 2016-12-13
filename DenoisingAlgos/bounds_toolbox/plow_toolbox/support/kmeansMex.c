/* compile with openMP flags in gcc, link with BLAS
 * For Unix architecture:
 * mex -largeArrayDims CC=gcc CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" kmeansMex.c -lmwblas 
 * For Windows:
 * mex -largeArrayDims kmeansMex.c <absolute_path_to_\libmwblas.lib>
 */


#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"
#include "blas.h"

#define	W_IN	prhs[0]
#define K_IN   prhs[1]

#define	IDX_OUT	plhs[0]
#define CEN_OUT   plhs[1]


double ddot(mwSignedIndex *, double *, mwSignedIndex *, double *, mwSignedIndex *);

int checkConvergence(double *, double *, mwSignedIndex);

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{ 
   mwSize mrows, ncols, midx, nidx, dims[2];
   int d,  iter=0, *clustSz, i, k, K, maxIter = 100, rnd; 
   double *Cen, *cenNorm, *newCen, dist[2], *W, *idx, One =1.0, csz, cn;
   mwSignedIndex inr=1, d2, dK;
           
   mrows = mxGetM(W_IN); /*Row size - size of each vector in features*/
   ncols = mxGetN(W_IN); /* Column size - number of vectors*/
   d = (int)mrows; d2 = d;
   
   /*printf("\n Recd %d features of size %d", ncols, mrows);*/
   
   /* Get input pointers */
   W = (double *)(mxGetPr(W_IN));
   K = (int)(mxGetScalar(K_IN));
   
  /* printf(" to cluster into %d parts", K);*/
   
   /* Create output arrays */
   /*dims[0] = 1; dims[1] = ncols;
   IDX_OUT = mxCreateNumericArray(2, dims, mxUINT32_CLASS, mxREAL)*/
   
   IDX_OUT = mxCreateDoubleMatrix(1,ncols,mxREAL);
   idx = (double *)(mxGetPr(IDX_OUT));
   
   CEN_OUT = mxCreateDoubleMatrix(1,(mwSize)K*d,mxREAL);
   Cen = (double *)(mxGetPr(CEN_OUT));
   
   dK = (mwSignedIndex)K*d2;
   newCen = (double *)malloc(dK*sizeof(double));
   cenNorm = (double *)malloc(K*sizeof(double));
   clustSz = (int *)malloc(K*sizeof(int));
   
  /* printf("\nDone initializing memory..");*/
      
   /* Populate Cen with random samples from input W matrix */
   /*randomize(); /* Generate random seed based on current time */
   for(k=0; k<K; k++){
       rnd = rand()%ncols; /* Select a random vector as center */
       dcopy(&d2,(W + rnd*d), &inr, (newCen + d*k), &inr); /* Copy Cen(k,:) <-- W(rnd,:) */
   }
   
   /*printf("\nDone initializing centers. Proceeding with batch iteration...");*/

   /* Batch update iteration */
 
   
   do {    
       
       /* Copy newCen to Cen */
       dcopy(&dK, newCen, &inr, Cen, &inr);
       
       
       /* Assign cluster to each point based on nearest centroid */
       
       /* Note that: for a point w, it belongs to c1 if for any c2
        *    ||y - c1||^2 < ||y - c2||^2
        * => ||c1||^2 - 2c1.y < ||c2||^2 - 2c2.y */
       
       /* Compute norm of all centers */
       for(k=0; k<K; k++){
           cn = dnrm2(&d2, Cen + k*d, &inr);/*ddot(&d2, Cen + k*d, &inr, Cen + k*d, &inr);*/
           cenNorm[k] = cn*cn;
           clustSz[k] = 0; /* Initialize cluster size to 0 */
        }
       /* Find cluster membership */
      /* printf("\n Find cluster membership...."); */
       for(i=0; i<ncols; i++){
           
           dist[0] = cenNorm[0] - 2*ddot(&d2, W + i*d, &inr, Cen, &inr);
           idx[i] = 0;
           
           for(k=1; k<K; k++) {
               dist[1] = cenNorm[k] - 2*ddot(&d2, W + i*d, &inr, Cen + k*d, &inr);
            
               if(dist[0] > dist[1]){
                   dist[0] = dist[1];
                   idx[i] = k;
               }
           }
           
           /* Now we know where point i lies, update information for new cluster center */
           /* newCen(k,:) = newCen(k,:) + W(i,:);*/
           daxpy(&d2, &One, W + i*d, &inr, newCen + (int)(idx[i])*d, &inr);
           clustSz[(int)(idx[i])]++;
           idx[i]++; /* Since Matlab counts from 1-K */
       } 
       
       /* Update cluster centers: newCen(k,:) = newCen(k,:)/clustSz */
       /*printf("[Done]\n Updating centers in iteration %d...", iter); */
        for(k=0; k<K; k++){
           csz = 1.0/((double)clustSz[k]);
           dscal(&d2, &csz, newCen + k*d, &inr);
        }
       /*printf("[Done]");*/
       /* Now to loop with new centers, only if the maxIter has not been exceeded
        * or cluster centers have shifted from the previous iteration */    
    
   } while( (++iter<maxIter) && (!checkConvergence(Cen, newCen, dK)));
   
   free(newCen);
   free(cenNorm);
   free(clustSz);
   
   /*printf("\nDone clustering in Mex\n");*/
}


/* Function to check if centers have changed beyond a tolerance 
 * Return true if converged, that is no change */
int checkConvergence(double *Cen, double *newCen, mwSignedIndex dK)
{
    double tol = 0.0; /* For testing purposes, let's keep this at 0 */
    double distMoved, negOne = -1.0;
    double *cenDiff = (double *)malloc(((int)dK)*sizeof(double));
    int i;
    mwSignedIndex inr = 1;
    
    /*dcopy(&dK, Cen, &inr, cenDiff, &inr);
    daxpy(&dk, &negOne, newCen, &inr, cenDiff, &inr);*/
    for(i=0; i<dK; i++)
        cenDiff[i] = Cen[i] - newCen[i];
    
    distMoved = dnrm2(&dK, cenDiff, &inr);
    free(cenDiff);
    
    return (distMoved <= tol)? 1: 0;
}
