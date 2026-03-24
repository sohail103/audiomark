/**
 * Comprehensive test for RVV DSP implementations
 * Tests: v_add_f32, v_subtract_f32, v_multiply_f32, v_dot_prod_f32
 * Run with: qemu-riscv64 -cpu rv64,v=true ./test_rvv_dsp
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef float ee_f32_t;

/* RVV implementations */
void v_add_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void v_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void v_multiply_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void v_dot_prod_f32(ee_f32_t *p_a, ee_f32_t *p_b, uint32_t len, ee_f32_t *p_result);

/* Scalar implementations */
void s_riscv_add_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void s_riscv_subtract_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void s_riscv_multiply_f32(ee_f32_t *p_a, ee_f32_t *p_b, ee_f32_t *p_c, uint32_t len);
void s_riscv_dot_prod_f32(ee_f32_t *p_a, ee_f32_t *p_b, uint32_t len, ee_f32_t *p_result);

#define EPSILON 1e-5f
#define TEST_SIZE_SMALL 4
#define TEST_SIZE_LARGE 256
#define TEST_SIZE_ODD 17

static int g_tests_passed = 0;
static int g_tests_failed = 0;

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
    printf("\n--- v_add_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_rvv[TEST_SIZE_SMALL], c_scalar[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        v_add_f32(a, b, c_rvv, TEST_SIZE_SMALL);
        s_riscv_add_f32(a, b, c_scalar, TEST_SIZE_SMALL);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_SMALL, "add small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_rvv[TEST_SIZE_LARGE], c_scalar[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        v_add_f32(a, b, c_rvv, TEST_SIZE_LARGE);
        s_riscv_add_f32(a, b, c_scalar, TEST_SIZE_LARGE);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_LARGE, "add large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_rvv[TEST_SIZE_ODD], c_scalar[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        v_add_f32(a, b, c_rvv, TEST_SIZE_ODD);
        s_riscv_add_f32(a, b, c_scalar, TEST_SIZE_ODD);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_ODD, "add odd-size");
    }
}

/* ============== SUBTRACT TESTS ============== */
static void test_subtract(void)
{
    printf("\n--- v_subtract_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_rvv[TEST_SIZE_SMALL], c_scalar[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        v_subtract_f32(a, b, c_rvv, TEST_SIZE_SMALL);
        s_riscv_subtract_f32(a, b, c_scalar, TEST_SIZE_SMALL);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_SMALL, "sub small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_rvv[TEST_SIZE_LARGE], c_scalar[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        v_subtract_f32(a, b, c_rvv, TEST_SIZE_LARGE);
        s_riscv_subtract_f32(a, b, c_scalar, TEST_SIZE_LARGE);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_LARGE, "sub large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_rvv[TEST_SIZE_ODD], c_scalar[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        v_subtract_f32(a, b, c_rvv, TEST_SIZE_ODD);
        s_riscv_subtract_f32(a, b, c_scalar, TEST_SIZE_ODD);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_ODD, "sub odd-size");
    }
}

/* ============== MULTIPLY TESTS ============== */
static void test_multiply(void)
{
    printf("\n--- v_multiply_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t c_rvv[TEST_SIZE_SMALL], c_scalar[TEST_SIZE_SMALL];
        init_test_data(a, b, TEST_SIZE_SMALL);

        v_multiply_f32(a, b, c_rvv, TEST_SIZE_SMALL);
        s_riscv_multiply_f32(a, b, c_scalar, TEST_SIZE_SMALL);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_SMALL, "mul small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t c_rvv[TEST_SIZE_LARGE], c_scalar[TEST_SIZE_LARGE];
        init_test_data(a, b, TEST_SIZE_LARGE);

        v_multiply_f32(a, b, c_rvv, TEST_SIZE_LARGE);
        s_riscv_multiply_f32(a, b, c_scalar, TEST_SIZE_LARGE);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_LARGE, "mul large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t c_rvv[TEST_SIZE_ODD], c_scalar[TEST_SIZE_ODD];
        init_test_data(a, b, TEST_SIZE_ODD);

        v_multiply_f32(a, b, c_rvv, TEST_SIZE_ODD);
        s_riscv_multiply_f32(a, b, c_scalar, TEST_SIZE_ODD);

        compare_arrays(c_scalar, c_rvv, TEST_SIZE_ODD, "mul odd-size");
    }
}

/* ============== DOT PRODUCT TESTS ============== */
static void test_dot_prod(void)
{
    printf("\n--- v_dot_prod_f32 Tests ---\n");

    /* Small array */
    {
        ee_f32_t a[TEST_SIZE_SMALL], b[TEST_SIZE_SMALL];
        ee_f32_t result_rvv, result_scalar;
        init_test_data(a, b, TEST_SIZE_SMALL);

        v_dot_prod_f32(a, b, TEST_SIZE_SMALL, &result_rvv);
        s_riscv_dot_prod_f32(a, b, TEST_SIZE_SMALL, &result_scalar);

        compare_scalar(result_scalar, result_rvv, "dot small");
    }

    /* Large array */
    {
        ee_f32_t a[TEST_SIZE_LARGE], b[TEST_SIZE_LARGE];
        ee_f32_t result_rvv, result_scalar;
        init_test_data(a, b, TEST_SIZE_LARGE);

        v_dot_prod_f32(a, b, TEST_SIZE_LARGE, &result_rvv);
        s_riscv_dot_prod_f32(a, b, TEST_SIZE_LARGE, &result_scalar);

        compare_scalar(result_scalar, result_rvv, "dot large");
    }

    /* Odd-sized array */
    {
        ee_f32_t a[TEST_SIZE_ODD], b[TEST_SIZE_ODD];
        ee_f32_t result_rvv, result_scalar;
        init_test_data(a, b, TEST_SIZE_ODD);

        v_dot_prod_f32(a, b, TEST_SIZE_ODD, &result_rvv);
        s_riscv_dot_prod_f32(a, b, TEST_SIZE_ODD, &result_scalar);

        compare_scalar(result_scalar, result_rvv, "dot odd-size");
    }

    /* Single element */
    {
        ee_f32_t a[] = {3.0f};
        ee_f32_t b[] = {4.0f};
        ee_f32_t result_rvv, result_scalar;

        v_dot_prod_f32(a, b, 1, &result_rvv);
        s_riscv_dot_prod_f32(a, b, 1, &result_scalar);

        compare_scalar(result_scalar, result_rvv, "dot single");
    }
}

int main(void)
{
    printf("========================================\n");
    printf("   RVV DSP Function Test Suite\n");
    printf("========================================\n");

#if defined(__riscv_vector) && defined(__riscv_zve32f)
    printf("RVV Mode: ENABLED (Zve32f detected)\n");
#else
    printf("RVV Mode: DISABLED (scalar fallback)\n");
#endif

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
