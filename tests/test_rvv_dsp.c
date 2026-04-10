/**
 * Comprehensive test for RISC-V DSP implementations
 * Tests: th_add_f32, th_subtract_f32, th_multiply_f32, th_dot_prod_f32
 * Run with:
 *   Scalar: qemu-riscv64 ./test_rvv_dsp
 *   Vector: qemu-riscv64 -cpu rv64,v=true ./test_rvv_dsp
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ee_api.h"

#define EPSILON 1e-5f
#define TEST_SIZE_SMALL 4
#define TEST_SIZE_LARGE 256
#define TEST_SIZE_ODD 17

static int g_tests_passed = 0;
static int g_tests_failed = 0;

static void
ref_add(const ee_f32_t *a, const ee_f32_t *b, ee_f32_t *c, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
    {
        c[i] = a[i] + b[i];
    }
}

static void
ref_subtract(const ee_f32_t *a, const ee_f32_t *b, ee_f32_t *c, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
    {
        c[i] = a[i] - b[i];
    }
}

static void
ref_multiply(const ee_f32_t *a, const ee_f32_t *b, ee_f32_t *c, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
    {
        c[i] = a[i] * b[i];
    }
}

static ee_f32_t
ref_dot_prod(const ee_f32_t *a, const ee_f32_t *b, uint32_t len)
{
    ee_f32_t sum = 0.0f;
    for (uint32_t i = 0; i < len; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

static int compare_arrays(const ee_f32_t *expected, const ee_f32_t *actual,
                          uint32_t len, const char *test_name)
{
    for (uint32_t i = 0; i < len; i++)
    {
        float diff = fabsf(expected[i] - actual[i]);
        float max_val = fmaxf(fabsf(expected[i]), fabsf(actual[i]));
        float rel_err = (max_val > EPSILON) ? diff / max_val : diff;
        
        if (rel_err > EPSILON)
        {
            printf("    FAIL at [%u]: expected %.6f, got %.6f (err=%.2e)\n",
                   i, expected[i], actual[i], rel_err);
            g_tests_failed++;
            return 0;
        }
    }
    printf("    %s: PASS\n", test_name);
    g_tests_passed++;
    return 1;
}

static int compare_scalar(ee_f32_t expected, ee_f32_t actual, const char *test_name)
{
    float diff = fabsf(expected - actual);
    float max_val = fmaxf(fabsf(expected), fabsf(actual));
    float rel_err = (max_val > EPSILON) ? diff / max_val : diff;

    if (rel_err > EPSILON)
    {
        printf("    FAIL: expected %.6f, got %.6f (err=%.2e)\n",
               expected, actual, rel_err);
        g_tests_failed++;
        return 0;
    }
    printf("    %s: PASS (result=%.6f)\n", test_name, actual);
    g_tests_passed++;
    return 1;
}

static void init_test_data(ee_f32_t *a, ee_f32_t *b, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
    {
        a[i] = (float)i * 0.1f + 1.0f;
        b[i] = (float)(len - i) * 0.05f + 0.5f;
    }
}

/* ============== ADD TESTS ============== */
static void test_add(void)
{
    printf("\n--- th_add_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_actual[TEST_SIZE_SMALL], c_expected[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        th_add_f32(a, b, c_actual, TEST_SIZE_SMALL);
        ref_add(a, b, c_expected, TEST_SIZE_SMALL);

        compare_arrays(c_expected, c_actual, TEST_SIZE_SMALL, "add small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_actual[TEST_SIZE_LARGE], c_expected[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        th_add_f32(a, b, c_actual, TEST_SIZE_LARGE);
        ref_add(a, b, c_expected, TEST_SIZE_LARGE);

        compare_arrays(c_expected, c_actual, TEST_SIZE_LARGE, "add large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_actual[TEST_SIZE_ODD], c_expected[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        th_add_f32(a, b, c_actual, TEST_SIZE_ODD);
        ref_add(a, b, c_expected, TEST_SIZE_ODD);

        compare_arrays(c_expected, c_actual, TEST_SIZE_ODD, "add odd-size");
    }
}

/* ============== SUBTRACT TESTS ============== */
static void test_subtract(void)
{
    printf("\n--- th_subtract_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_actual[TEST_SIZE_SMALL], c_expected[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        th_subtract_f32(a, b, c_actual, TEST_SIZE_SMALL);
        ref_subtract(a, b, c_expected, TEST_SIZE_SMALL);

        compare_arrays(c_expected, c_actual, TEST_SIZE_SMALL, "sub small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_actual[TEST_SIZE_LARGE], c_expected[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        th_subtract_f32(a, b, c_actual, TEST_SIZE_LARGE);
        ref_subtract(a, b, c_expected, TEST_SIZE_LARGE);

        compare_arrays(c_expected, c_actual, TEST_SIZE_LARGE, "sub large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_actual[TEST_SIZE_ODD], c_expected[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        th_subtract_f32(a, b, c_actual, TEST_SIZE_ODD);
        ref_subtract(a, b, c_expected, TEST_SIZE_ODD);

        compare_arrays(c_expected, c_actual, TEST_SIZE_ODD, "sub odd-size");
    }
}

/* ============== MULTIPLY TESTS ============== */
static void test_multiply(void)
{
    printf("\n--- th_multiply_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_actual[TEST_SIZE_SMALL], c_expected[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        th_multiply_f32(a, b, c_actual, TEST_SIZE_SMALL);
        ref_multiply(a, b, c_expected, TEST_SIZE_SMALL);

        compare_arrays(c_expected, c_actual, TEST_SIZE_SMALL, "mul small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_actual[TEST_SIZE_LARGE], c_expected[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        th_multiply_f32(a, b, c_actual, TEST_SIZE_LARGE);
        ref_multiply(a, b, c_expected, TEST_SIZE_LARGE);

        compare_arrays(c_expected, c_actual, TEST_SIZE_LARGE, "mul large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_actual[TEST_SIZE_ODD], c_expected[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        th_multiply_f32(a, b, c_actual, TEST_SIZE_ODD);
        ref_multiply(a, b, c_expected, TEST_SIZE_ODD);

        compare_arrays(c_expected, c_actual, TEST_SIZE_ODD, "mul odd-size");
    }
}

/* ============== DOT PRODUCT TESTS ============== */
static void test_dot_prod(void)
{
    printf("\n--- th_dot_prod_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t result_actual;
        ee_f32_t result_expected;
        init_test_data(a, b, TEST_SIZE_SMALL);

        th_dot_prod_f32(a, b, TEST_SIZE_SMALL, &result_actual);
        result_expected = ref_dot_prod(a, b, TEST_SIZE_SMALL);

        compare_scalar(result_expected, result_actual, "dot small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t result_actual;
        ee_f32_t result_expected;
        init_test_data(a, b, TEST_SIZE_LARGE);

        th_dot_prod_f32(a, b, TEST_SIZE_LARGE, &result_actual);
        result_expected = ref_dot_prod(a, b, TEST_SIZE_LARGE);

        compare_scalar(result_expected, result_actual, "dot large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t result_actual;
        ee_f32_t result_expected;
        init_test_data(a, b, TEST_SIZE_ODD);

        th_dot_prod_f32(a, b, TEST_SIZE_ODD, &result_actual);
        result_expected = ref_dot_prod(a, b, TEST_SIZE_ODD);

        compare_scalar(result_expected, result_actual, "dot odd-size");
    }

    /* Single element */
    {
        ee_f32_t a[] = {3.0f};
        ee_f32_t b[] = {4.0f};
        ee_f32_t result_actual;
        ee_f32_t result_expected;

        th_dot_prod_f32(a, b, 1, &result_actual);
        result_expected = ref_dot_prod(a, b, 1);

        compare_scalar(result_expected, result_actual, "dot single");
    }
}

int main(void)
{
    printf("========================================\n");
    printf("   RVV DSP Function Test Suite\n");
    printf("========================================\n");

    printf("Validates active port implementation against scalar references.\n");

    test_add();
    test_subtract();
    test_multiply();
    test_dot_prod();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("========================================\n");

    if (g_tests_failed == 0)
    {
        printf("ALL TESTS PASSED!\n");
        return 0;
    }
    else
    {
        printf("SOME TESTS FAILED!\n");
        return 1;
    }
}
