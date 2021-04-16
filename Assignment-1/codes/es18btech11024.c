#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

/*
Program to calculate (Fourier Transform)
using the vector-radix FFT algorithm.
*/

void fft(double complex *a, long int N, int temp)
{
    if (N == 1)
        return;

    double complex *Odd = malloc(N / 2 * sizeof(double complex));
    double complex *Even = malloc(N / 2 * sizeof(double complex));
    for (long int i = 0; i < N / 2; i++)
    {
        Odd[i] = a[2 * i + 1];
        Even[i] = a[2 * i];
    }

    fft(Even, N / 2, temp);
    fft(Odd, N / 2, temp);

    for (int i = 0; i < N / 2; i++)
    {
        double temp1 = M_PI * i * I * temp / N;
        double complex x = exp(2 * temp1); //Twiddle Factor Cal part
        a[i] = Even[i] + x * Odd[i];
        a[i + N / 2] = Even[i] - x * Odd[i];
    }

    return;
}

void ifft(double complex *a, long int N)
{

    fft(a, N, -1);

    for (int i = 0; i < N; i++)
        a[i] = a[i] / N;

    return;
}

int main()
{

    int n = (1 << 20);

    double *a1 = (double *)malloc(n * sizeof(double));
    double *a2 = (double *)malloc(n * sizeof(double));

    double complex *A1 = (double complex *)malloc(n * sizeof(double complex));
    double complex *A2 = (double complex *)malloc(n * sizeof(double complex));

    FILE *FT1, *FF1;

    FT1 = fopen("Sound_Noise.dat", "r");

    int length = 0;

    while (!feof(FT1) && length < n)
    {
        fscanf(FT1, "%1f", &(a1[length]));
        A1[length] = CMPLX(a1[length], 0);
        length++;
    }

    fft(A1, n, 1);

    FF1 = fopen("A1.dat", "w");
    for (int i = 0; i < n; i++)
        fprintf(FF1, "%1f+%1fi\n", creal(A1[i]), cimag(A1[i]));

    fclose(FT1);

    fclose(FF1);
    return 0;
}