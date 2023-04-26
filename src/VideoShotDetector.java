public class VideoShotDetector {

/*
*   Algorithm Pseudocode:
*       1: break video into segments, S_i, of 25 frames: S_1, S_2, S_3, ..., S_n
*       2: GOAL: For each segment S_i, determine whether the segment is 'static' or 'dynamic'
*           - Take the first frame and last frame of S_i: f_i,1 and f_i,n
*           - Break the frames into a 3x3 grid of blocks
*           - for the pixels in each block, calculate the Y (luminance), R, G, and B histograms
*           - concatenate all blocks together to form h_i
*                               !!!The concatenation of the histograms of the nine blocks
*                                   represents the CBBH of the frame fi, noted hi. (used in next stage)
*           - Calculate ~h_i: center h_i by subtracting the column mean from all values in the column,
*                             for each column
*           - calculate the correlation coefficient, d_i,j using ~h_i,1 and ~h_i,25
*               - d_i,j = (inner product of ~h_i,1, ~h_i,25) / (matrix norm of ~h_i,1 * matrix norm of ~h_i,25)
*           - if d_i,j is greater than a threshold T_s, then the segment is 'static', if not, it is 'dynamic'
*
*       3: Feature Construction (for dynamic segments)
*           - Construct feature matrix H = [h1, h2, h3, ... h_n], where n is the segment length and
*                   each h_i is a frame's CBBH (calculated in step 2)
*           - GOAL: Find a ~k value that retains all important information of the frames
*                      In other words, keeping only the k-largest singular values is the same
*                       as keeping only the relevant information of a scene.
*               - calculate Sigma = diag(σ1,σ2,...,σn).
*               - ~k is in the range of [1, r], where r = rank(H)
*               - iterate the value k from 1 to r until it satisfies:
*                       Sum(i=k+1 to r: σi^2) < ε / (1-ε) * Sum(i=1 to k: σi^2)
*               - ~k = k
*               -each column hi will be mapped into the singular space and represented
*                   with a reduced projected vector [phi]_i ∈ R^k ̃ according to the matrix
*
*
*
* */


}
