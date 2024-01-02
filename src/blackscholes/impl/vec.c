/* vec.c
 *
 * Author:
 * Date  :
 *
 *  Description
 */

/* Standard C includes */
#include <stdlib.h>
#include <stdio.h>

/* Include common headers */
#include "common/macros.h"
#include "common/types.h"

/* Include application-specific headers */
#include "include/types.h"
#define inv_sqrt_2xPI 0.39894228040143270286
#include <math.h>
#include <immintrin.h>
#include <string.h>

#include "scalar.h"

__m256 exp256_ps(__m256 x) {
  float *values = (float*)&x;
  for (int i = 0; i < 8; i++) {
    values[i] = exp(values[i]);
  }
  return x;
}

__m256 log256_ps(__m256 x) {
  float *values = (float*)&x;
  for (int i = 0; i < 8; i++) {
    values[i] = log(values[i]);
  }
  return x;
}

__m256 make_vec_of(float value) 
{
  return _mm256_setr_ps(value, value, value, value, value, value, value, value);
}

__m256i make_veci_of(int value) 
{
  return _mm256_setr_epi32(value, value, value, value, value, value, value, value);
}

float get_first(__m256 vec) {
  return ((float*)&vec)[0];
}

__m256 vectorCNDF(__m256 inputs)
{ 
  __m256 negative_ones = make_vec_of(-1.0f);
  __m256 zeros = make_vec_of(0.0f);
  __m256 neg_halves = make_vec_of(-0.5f);
  __m256 ones = make_vec_of(1.0f);
  __m256 inv_sqrt_2xPIs = make_vec_of(inv_sqrt_2xPI);
  __m256 magic_number_1s = make_vec_of(0.2316419);
  __m256 magic_number_2s = make_vec_of(0.319381530);

  __m256 signs;
  __m256 abs_inputs;
  __m256 exp_values;
  __m256 n_prime_of_xs;
  __m256 x_k2;
  __m256 x_k2_2;
  __m256 x_k2_3;
  __m256 x_k2_4;
  __m256 x_k2_5;
  __m256 x_local;
  __m256 x_local1;
  __m256 x_local2;
  __m256 x_local3;
  __m256 alternative_if_negative;

  signs =  _mm256_cmp_ps (inputs, zeros, 17); // OP := _CMP_LT_OQ
  abs_inputs = _mm256_mul_ps(inputs, _mm256_blendv_ps(ones, negative_ones, signs));
  
  exp_values = exp256_ps(_mm256_mul_ps(neg_halves, _mm256_mul_ps(abs_inputs, abs_inputs)));
  n_prime_of_xs = _mm256_mul_ps(exp_values, inv_sqrt_2xPIs);
  x_k2 = _mm256_mul_ps(magic_number_1s, abs_inputs);
  x_k2 = _mm256_add_ps(ones, x_k2);
  x_k2 = _mm256_div_ps(ones, x_k2);
  x_k2_2 = _mm256_mul_ps(x_k2, x_k2);
  x_k2_3 = _mm256_mul_ps(x_k2_2, x_k2);
  x_k2_4 = _mm256_mul_ps(x_k2_3, x_k2);
  x_k2_5 = _mm256_mul_ps(x_k2_4, x_k2);
 
  x_local1 = _mm256_mul_ps(x_k2, make_vec_of(0.319381530f));
  x_local2 = _mm256_mul_ps(x_k2_2, make_vec_of(-0.356563782f));
  x_local3 = _mm256_mul_ps(x_k2_3, make_vec_of(1.781477937f));
  x_local2 = _mm256_add_ps(x_local2, x_local3);
  x_local3 = _mm256_mul_ps(x_k2_4, make_vec_of(-1.821255978f));
  x_local2 = _mm256_add_ps(x_local2, x_local3);
  x_local3 = _mm256_mul_ps(x_k2_5, make_vec_of(1.330274429f));
  x_local2 = _mm256_add_ps(x_local2, x_local3);
  x_local1 = _mm256_add_ps(x_local2, x_local1);
  x_local = _mm256_mul_ps(x_local1, n_prime_of_xs);
  x_local = _mm256_sub_ps(ones, x_local);
  

  alternative_if_negative = _mm256_sub_ps(ones, x_local);
  __m256 result = _mm256_blendv_ps(x_local, alternative_if_negative, signs);

  return result;
}

__m256 load_vec_from(float *values) 
{
  return _mm256_setr_ps(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
}

__m256i load_veci_from(int *values) 
{
  return _mm256_setr_epi32(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
}

void vectorBlackScholes(float *sptprice, float *strike, float *rate, float *volatility, float *otime, int *otype, float *out)
{
  // local private working variables for the calculation
  __m256 xStockPrice;
  __m256 xStrikePrice;
  __m256 xRiskFreeRate;
  __m256 xVolatility;
  __m256 xTime;
  __m256 xSqrtTime;
  __m256 logValues;
  __m256 xLogTerm;
  __m256 xD1;
  __m256 xD2;
  __m256 xPowerTerm;
  __m256 xDen;
  __m256 d1;
  __m256 d2;
  __m256 FutureValueX;
  __m256 NofXd1;
  __m256 NofXd2;
  __m256 NegNofXd1;
  __m256 NegNofXd2;
  __m256i comparasion;
  __m256 if_not_zero;
  __m256 if_zero;
 
  xStockPrice = load_vec_from(sptprice);
  xStrikePrice = load_vec_from(strike);
  xRiskFreeRate = load_vec_from(rate);
  xVolatility = load_vec_from(volatility);
  xTime = load_vec_from(otime);
 
  xSqrtTime = _mm256_sqrt_ps(xTime);
  logValues = log256_ps(_mm256_div_ps(xStockPrice, xStrikePrice));
  xLogTerm = logValues; 
 
  xPowerTerm = _mm256_mul_ps(xVolatility, xVolatility);
  xPowerTerm = _mm256_mul_ps(xPowerTerm, make_vec_of(0.5f));
 
  xD1 = _mm256_add_ps(xRiskFreeRate, xPowerTerm);
  xD1 = _mm256_mul_ps(xD1, xTime);
  xD1 = _mm256_add_ps(xD1, xLogTerm);
  xDen = _mm256_mul_ps(xVolatility, xSqrtTime);
  xD1 = _mm256_div_ps(xD1, xDen);
  xD2 = _mm256_sub_ps(xD1, xDen);
 
  d1 = xD1;
  d2 = xD2;
 
  NofXd1 = vectorCNDF(d1);
  NofXd2 = vectorCNDF(d2);
  FutureValueX = _mm256_mul_ps(xStrikePrice, exp256_ps(_mm256_mul_ps(_mm256_mul_ps(xRiskFreeRate, xTime), make_vec_of(-1.0f))));
  comparasion = _mm256_cmpeq_epi32(load_veci_from(otype), make_veci_of(0));
  if_zero = _mm256_sub_ps(_mm256_mul_ps(xStockPrice, NofXd1), _mm256_mul_ps(FutureValueX, NofXd2));
  NegNofXd1 = _mm256_sub_ps(make_vec_of(1.0f), NofXd1);
  NegNofXd2 = _mm256_sub_ps(make_vec_of(1.0f), NofXd2);
  if_not_zero = _mm256_sub_ps(_mm256_mul_ps(FutureValueX, NegNofXd2), _mm256_mul_ps(xStockPrice, NegNofXd1));
  xStockPrice = _mm256_blendv_ps(if_not_zero, if_zero, _mm256_castsi256_ps(comparasion));

  memcpy(out, (float*)&xStockPrice, sizeof(float) * 8);
 }
 
 //

/* Alternative Implementation */ 
void* impl_vector(void* args)
{
  args_t* data = (args_t*)args;

  int iteration_count_for_vectorized = data->num_stocks / 8;
  int remaining_for_scalar = data->num_stocks % 8;
  
  for (int i = 0; i < iteration_count_for_vectorized; i++) {
    int index = i * 8;
    float* sptprice = data->sptPrice + index;
    float* strike = data->strike + index;
    float* rate = data->rate + index;
    float* volatility = data->volatility + index;
    float* otime = data->otime + index;
    int otypes[8];
    for (int j = 0; j < 8; j++) {
      otypes[j] = (data->otype[j + index] == 'P') ? 1 : 0;
    }

    vectorBlackScholes(sptprice, strike, rate, volatility, otime, otypes, data->output + index);
  }

  for (int i = iteration_count_for_vectorized * 8; i < data->num_stocks; i++) {
    float sptprice = data->sptPrice[i];
    float strike = data->strike[i];
    float rate = data->rate[i];
    float volatility = data->volatility[i];
    float otime = data->otime[i];
    int otype = (data->otype[i] == 'P') ? 1 : 0; 

    // Perform the Black-Scholes calculation
    float price = blackScholes(sptprice, strike, rate, volatility, otime, otype, 0.0);
    // Store the result in the destination array
    data->output[i] = price;
  }

  return NULL;
}

void add(int *a, int *b, int *out, int length) {
  for (int i = 0; i < length; i++) {
    out[i] = a[i] + b[i];
  }
}

/*
    __m256i vm = _mm256_set_epi32(m[0], m[1], m[2], m[3],
                             m[4], m[5], m[6], m[7]);


 */ 
// assume length % 8 == 0
void vector_add(int *a, int *b, int *out, int length) {
  for (int i = 0; i < length; i += 8) { 
    __m256i vec_a = _mm256_setr_epi32(a[0 + i], a[1 + i], a[2 + i], a[3 + i], a[4 + i], a[5 + i], a[6 + i], a[7 + i]);
    __m256i vec_b = _mm256_setr_epi32(b[0 + i], b[1 + i], b[2 + i], b[3 + i], b[4 + i], b[5 + i], b[6 + i], b[7 + i]);
    __m256i output = _mm256_add_epi32(vec_a, vec_b); //is it like this?
    memcpy(out + i, (float*)&output, sizeof(int) * 8);
  }
}

/*
Ways to use SIMD for code speedup
1) 
int add_a_lot(int a, int b, int c, int d) {
  return a + b + c + d; // = 3 operations 
}
->
int add_a_lot(int a, int b, int c, int d) {
  __simd2_4int result_1 = __simd_add(a, b, c, d); // (a + b, c + d)
  return __simd_add(result_1); // (result_1[0] + result_1[1]);
}

2)
void add_a_lot(int *a, int *b, int *c, int *d, int *out, int length) {
  for (int i = 0; i < length; i++) {
    
  }
}



vectors_size = 1
a = {0, 1, 2, 3}
b = {9, 8, 7, 6}
o_0 = a_0 + b_0
o_1 = a_1 + b_1
o_2 = a_2 + b_2
o_4 = a_3 + b_3

o = a + b

*/