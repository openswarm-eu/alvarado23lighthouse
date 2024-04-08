/**
 * @file
 * @ingroup LH_Location
 * @author Said Alvarado-Marin <said-alexander.alvarado-marin@inria.fr>
 * @brief Benchmark test of the twoLH_2D algorithm
 *
 *
 * @date 2024
 *
 * @copyright Inria, 2024
 *
 */
#include <nrf.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "board.h"
#include "board_config.h"
#include "lh2.h"

//=========================== defines ==========================================

//=========================== variables ========================================

//=========================== prototypes =======================================

void calculate_camera_point(uint32_t count1, uint32_t count2, uint8_t poly_in, float *cam_x, float *cam_y);

void multiplyMatrixByVector(float matrix[3][3], float vector[3], float result[3]);

void twoLH_2D(float R[3][3], float t[3], float n_star[3], float zeta, float cam_pts[2], float result[3]);

//=========================== main =============================================

/**
 *  @brief The program starts executing here.
 */
int main(void) {
    // Initialize the board core features (voltage regulator)
    db_board_init();

    float cam_x, cam_y;
    uint32_t count1 = 44807;
    uint32_t count2 = 88762;
    uint8_t poly_in = 1;

    // Define the 3x3 matrix
    float A[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    // Define the 3x1 vector
    float v[3] = {0.3, -0.45, 1.0}; // Replace x, y, z with actual values
    // Define the result vector
    float result[3];
    float input[2];

    NRF_P0->DIRSET = 1 << 24;
    NRF_P1->DIRSET = 1 << 10;
    NRF_P1->DIRSET = 1 << 13;

    while (1) {
        // the location function has to be running all the time

        NRF_P0->OUTSET  = 1 << 24;
        calculate_camera_point(count1, count2, poly_in, &cam_x, &cam_y);
        NRF_P0->OUTCLR  = 1 << 24;

        // Perform the multiplication
        NRF_P1->OUTSET  = 1 << 10;
        input[0] = cam_x;
        input[1] = cam_y;
        twoLH_2D(A, v, v, 0.5, input, result);
        NRF_P1->OUTCLR  = 1 << 10;

    }

    // one last instruction, doesn't do anything, it's just to have a place to put a breakpoint.
    __NOP();
}

//=========================== private ==========================================

void calculate_camera_point(uint32_t count1, uint32_t count2, uint8_t poly_in, float *cam_x, float *cam_y) {
    uint32_t period;
    float a1, a2;

    // Determine the period based on poly_in
    period = poly_in < 2 ? 959000 : 957000;

    // Calculate angles
    a1 = (count1 * 8.0 / period) * 2 * M_PI;
    a2 = (count2 * 8.0 / period) * 2 * M_PI;

    // Calculate cam_x and cam_y
    *cam_x = -tan(0.5 * (a1 + a2));
    *cam_y = -sin(fabs(a2 - a1) / 2.0 - 1.0471975511965976) / 0.5773502691896257;
    // *cam_y = -sin(fabs(a2 - a1) / 2.0 - 60 * M_PI / 180) / tan(M_PI / 6);
}

void multiplyMatrixByVector(float matrix[3][3], float vector[3], float result[3]) {
    for (int i = 0; i < 3; i++) {
        result[i] = 0; // Initialize result element to 0
        for (int j = 0; j < 3; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void twoLH_2D(float R[3][3], float t[3], float n_star[3], float zeta, float cam_pts[2], float result[3]) {

    float hom_pts[3] = {cam_pts[0], cam_pts[1], 1.0};

    float scale = (1/zeta) / (n_star[0]*hom_pts[0] + n_star[1]*hom_pts[1] + n_star[2]);

    float final_point[3] = {scale*hom_pts[0], scale*hom_pts[1], scale*hom_pts[2]};

    float rot_pts[3];
    multiplyMatrixByVector(R, final_point, rot_pts);
    
    // translate and output
    for (size_t i = 0; i < 3; i++)
    {
        result[i] = rot_pts[i] + t[i];
    }
    
}