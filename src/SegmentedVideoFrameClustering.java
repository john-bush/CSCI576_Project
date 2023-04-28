import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;
import java.lang.Math;
import javax.imageio.ImageIO;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.RealMatrix;
import org.apache.commons.math4.legacy.linear.SingularValueDecomposition;

public class SegmentedVideoFrameClustering {
    private static final DecimalFormat df = new DecimalFormat("00.00");
    private static List<BufferedImage> frames = new ArrayList<>();

    static final int num_block_rows = 5;
    static final int num_block_cols = 5;
    static final int num_histograms = 4;
    static final int num_bins = 256;
    private static final int segment_length = 25; // number of frames in segment
    private static final int frame_step = 4; // frame step used for cut verification

//    THRESHOLDS AND TUNING PARAMETERS
    static final double epsilon = 0.0001;
    static final double TC1 = 0.75;
    static final double TC2 = 0.90;
    static final double correlation_threshold = 0.96;
    static final int DEBUG = 0; // to show correlation differences
    public static void main(String[] args) {
        File file = new File("lib/The_Great_Gatsby_rgb/InputVideo.rgb"); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video

        System.out.println("Blocks:"+num_block_rows+"x"+num_block_cols+", Thresholds: "+TC1+", " + TC2 + ", Epsilon: " + epsilon);

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);

            for (int i = 0; i < numFrames-1; i++) {
                buffer.clear();
                channel.read(buffer);
                buffer.rewind();
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                frames.add(image);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        image.setRGB(x, y, rgb);
                    }
                }
            }

            // calculate segment beginning-end frame correlations
            for (int i = 0; i < numFrames - segment_length - 1; i+=(segment_length - 1)) {
                // compute correlation between first and last frames of segment
                double correlation = computeCorrelation(i, i + segment_length - 1);

                if (DEBUG > 2) {
                    System.out.println("Segment Correlation: " + correlation);
                }

                if (correlation <= correlation_threshold) { // segment is dynamic
//      perform cut verification as covered in the paper
                    int rel_frame = cutTransitionID_Incremental(i);
                    if (rel_frame != -1) {
                        if (DEBUG == 2) {
                            System.out.print("Segment Frame: " + i + ", Local Frame: " + rel_frame + ", Shot: ");
                        }
                        int cut_frame = i + rel_frame;

                        // Start a new group
                        int minutes = ((cut_frame / 30) % 3600) / 60;
                        double seconds = (cut_frame / 30.0) % 60;
                        System.out.println(minutes + " min : " + df.format(seconds) + " secs");
                    }
                    //System.out.println(distance/10000);
                } // else { // segment is static, do nothing
            }

            try {
                Thread.sleep(1000 / fps);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            channel.close();
            raf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param index - frame index of start of segment to analyze
     * @return frame number of cut transition (-1 if the segment is static)
     */
    private static int cutTransitionID(int index) {
        double[][] matrixBeta = calculateMatrixBeta(index);
        int cut_index = -1;
        // look for distance less than TC1
            // if a distance is less than TC1 is found, calculate the distances between all pairs of frames
            // within that frame step. The frame pair with a distance less than TC1 is where the cut frame is

//        TODO: these for loop bounds are super sketchy
        // calculate distances on frame step
        double[] distances = new double[segment_length / (frame_step + 1)];
        if (DEBUG > 0){
            System.out.print(", Distances: [");
        }
        for (int d = 0; d < segment_length / (frame_step + 1); d++) {
            double[] frame1 = matrixBeta[d * frame_step];
            double[] frame2 = matrixBeta[d * frame_step + frame_step];
            distances[d] = computeCorrelation(frame1, frame2);
            if (DEBUG > 0) {
                System.out.print(distances[d] + ", ");
            }

            if (distances[d] < TC1) {
                double min_distance = 100.0;
                for (int i = d * frame_step; i < (d * frame_step + frame_step); i++) {
                    double correlation = computeCorrelation(matrixBeta[i], matrixBeta[i+1]);
                    if (correlation < min_distance) {
                        min_distance = correlation;
                        cut_index = d * frame_step + i + 1;
                    }
                }
                return cut_index;
            }
        }
        if (DEBUG > 0) {
            System.out.println("]");
        }

        // if all distances > TC1, check for a distance less than TC2
        // if there is only 1 distance < TC2, perform cut localization on the frames within that step.
        // if all distances > TC2, declare the segment static
        // if more than 1 distance < TC2, do the dynamic verification part of the paper (not written now)
        int threshold_count = 0;
        int threshold_index = -1;
        double lowest_dist = 5.0;
        for (int i = 0; i < distances.length; i++) {
            if (distances[i] < TC2) {
                threshold_count++;

                if (distances[i] < lowest_dist) {
                    lowest_dist = distances[i];
                    threshold_index = i;
                }
            }
        }
        if (threshold_count == 0) {
            return -1; // segment is static
        } else {
            double min_distance = 100.0;
            for (int i = threshold_index * frame_step; i < (threshold_index * frame_step + frame_step); i++) {
                double correlation = computeCorrelation(matrixBeta[i], matrixBeta[i + 1]);
                if (correlation < min_distance) {
                    min_distance = correlation;
                    cut_index = i + 1;
                }
            }
            return cut_index;
        }
    }

    private static int cutTransitionID_Incremental(int index) {
        double[][] matrixBeta = calculateMatrixBeta(index);
        int cut_index = -1;
        // look for distance less than TC1
        // if a distance is less than TC1 is found, calculate the distances between all pairs of frames
        // within that frame step. The frame pair with a distance less than TC1 is where the cut frame is

//        TODO: these for loop bounds are super sketchy
        // calculate distances on frame step
        double[] distances = new double[segment_length - (frame_step + 1)]; // 1 2 3 4 5 6 7 8 9 10
        if (DEBUG > 0){
            System.out.print(", Distances: [");
        }
        for (int d = 0; d < distances.length; d++) {
            double[] frame1 = matrixBeta[d];
            double[] frame2 = matrixBeta[d + frame_step + 1];
            distances[d] = computeCorrelation(frame1, frame2);
            if (DEBUG > 0) {
                System.out.print(distances[d] + ", ");
            }

            if (distances[d] < TC1) {
                double min_distance = 100.0;
                for (int i = d; i < (d + frame_step + 1); i++) {
                    double correlation = computeCorrelation(matrixBeta[i], matrixBeta[i+1]);
                    if (correlation < min_distance) {
                        min_distance = correlation;
                        cut_index = i + 1;
                    }
                }
                return cut_index;
            }
        }
        if (DEBUG > 0) {
            System.out.println("]");
        }

        // if all distances > TC1, check for a distance less than TC2
        // if there is only 1 distance < TC2, perform cut localization on the frames within that step.
        // if all distances > TC2, declare the segment static
        // if more than 1 distance < TC2, do the dynamic verification part of the paper (not written now)
        int threshold_count = 0;
        int threshold_index = -1;
        double lowest_dist = 5.0;
        for (int i = 0; i < distances.length; i++) {
            if (distances[i] < TC2) {
                threshold_count++;

                if (distances[i] < lowest_dist) {
                    lowest_dist = distances[i];
                    threshold_index = i;
                }
            }
        }
        if (threshold_count == 0) {
            return -1; // segment is static
        } else {
            double min_distance = 100.0;
            for (int i = threshold_index; i < (threshold_index + frame_step + 1); i++) {
                double correlation = computeCorrelation(matrixBeta[i], matrixBeta[i + 1]);
                if (correlation < min_distance) {
                    min_distance = correlation;
                    cut_index = i + 1;
                }
            }
            return cut_index;
        }
    }
/**
 *       3: Feature Construction (for dynamic segments)
 *           - Construct feature matrix H = [h1, h2, h3, ... h_n], where n is the segment length and
 *                   each h_i is a frame's CBBH (calculated in step 2) with length m
 *           - GOAL: Find a ~k value that retains all important information of the frames
 *                      In other words, keeping only the k-largest singular values is the same
 *                       as keeping only the relevant information of a scene.
 *               - calculate Sigma = diag(σ1,σ2,...,σn).
 *               - ~k is in the range of [1, r], where r = rank(H). The number of non-zero σ is the rank of H.
 *               - iterate the value k from 1 to r until it satisfies:
 *                       Sum(i=k+1 to r: σi^2) < ε / (1-ε) * Sum(i=1 to k: σi^2)
 *               - ~k = k
 *               -each column hi will be mapped into the singular space and represented
 *                   with a reduced projected vector [phi]_i ∈ R^k ̃ according to the matrix
 */


    /**
     * CONSTRUCT_MATRIX_H
     * @param index - index of start of segment
     * @return - matrix H of size [m][segment_length]
     */
    private static double[][] constructMatrixH(int index) {
        // length of each column vector
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;

        // construct matrix H by joining CBBH column vector of each frame
        double[][] H = new double[m][segment_length];

        // calculates each frame histogram and sets a column of H
        for (int i = 0; i < segment_length; i++) {
            List<Integer> column = computeHistogram(frames.get(i + index));
            for (int r = 0; r < m; r++) {
                H[r][i] = column.get(r);
            }
        }

        return H;
    }

    private static int findK(double[][] Sigma, int r) {
        // calculate the rank of H with Sigma
        int width = Sigma[0].length;
//        int r = 0;
//
//        for (int i = 0; i < width; i++) {
//            if (Sigma[i][i] > 0) {
//                r++;
//            } else {
//                break;
//            }
//        }
//        iterate the value k from 1 to r until it satisfies:
//           Sum(i=k+1 to r: σi^2) < ε / (1-ε) * Sum(i=1 to k: σi^2)
//        TODO: check array bounds on these for loops
        int k;
        for (k = 1; k < r; k++) {
            double sum1 = 0;
            double sum2 = 0;
            for (int i1 = k; i1 < r; i1++) {
                sum1 = sum1 + Sigma[i1][i1] * Sigma[i1][i1];
            }
            for (int i2 = 0; i2 < k; i2++) {
                sum2 = sum2 + Sigma[i2][i2] * Sigma[i2][i2];
            }
            sum2 = sum2 * ((epsilon)/(1 - epsilon));

            if (sum1 < sum2) {
                break;
            }
        }
        if (DEBUG > 2) {
            System.out.print(", Rank = " + r + ", K = " + k);
        }
        return k;
    }

    private static double[][] calculateMatrixBeta(int index) {
        int minutes = ((index / 30) % 3600) / 60;
        double seconds = (index / 30.0) % 60;
        if (DEBUG > 0) {
            System.out.print("Processing segment starting at " + minutes + ":" + df.format(seconds) + "s");

        }

        // length of each column vector
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;
        final int n = segment_length;

        // construct matrix H by joining CBBH column vector of each frame
        double[][] H = constructMatrixH(index);

        // create SVD object
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(H));

        double[][] Sigma = svd.getS().getData();

        int rank = svd.getRank();

        int k = findK(Sigma, rank);

        double[][] truncatedS = new double[k][k];
        svd.getS().copySubMatrix(0, k - 1, 0, k - 1, truncatedS);

        double[][] truncatedVT = new double[k][svd.getVT().getColumnDimension()];
        svd.getVT().copySubMatrix(0, k - 1, 0, truncatedVT[0].length - 1, truncatedVT);

        double[][] truncatedU = new double[svd.getU().getRowDimension()][k];
        svd.getU().copySubMatrix(0, truncatedU.length - 1, 0, k - 1, truncatedU);


//        NOTE: i am representing this matrix transposed to how its represented in the paper
//              because it allows us to extract a single B vector as a row of B
//        double[][] Beta_matrix = new double[segment_length][k]; // initialize result vector of length k


        RealMatrix approximatedSvdMatrix = (new Array2DRowRealMatrix(truncatedU)).multiply(new Array2DRowRealMatrix(truncatedS)).multiply(new Array2DRowRealMatrix(truncatedVT)).transpose();

        return approximatedSvdMatrix.getData();
        // calculate the B_matrix
//        for (int b_row = 0; b_row < segment_length; b_row++) {
//            for (int i = 0; i < k; i++) {
//                Beta_matrix[b_row][i] += truncatedS[i][i] * truncatedVT[i][b_row];
//            }
//        }

//        return Beta_matrix;
    }

    /**
     *
     * @param index - first frame of segment, with length segment_length
     * @return cut_index : index of the cut frame (absolute index, not relative to i)
     */
    private static int findCutFrame(int index) {
        double lowest_correlation = 200.0;
        int low_correlation_idx = index + segment_length;

        for (int i = index; i < index + segment_length - 1; i++) {
            double correlation = computeCorrelation(i, i+1);

            // calculate lowest correlation value within the segment
            if (correlation < lowest_correlation) {
                lowest_correlation = correlation;
                low_correlation_idx = i+1; // set frame cut index to the second frame in the comparison
            }
        }

        return low_correlation_idx;
    }

    /**
     * @function computeHistogram(BufferedImage image)
     * @param image - rgb buffered image
     * @return calculates a concatenated block histogram (CBBH) of the image.
     *          breaks the image into 3x3 blocks and calculates the R, G, B and Y
     *          histograms for each block.
     *          (each of the 9 block's histograms are all concatenated together, such as:
     *              [B1R, B1G, B1B, B1Y, B2R, B2G, B2B, B2Y, ...]
     */


    private static List<Integer> computeHistogram(BufferedImage image) {
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;
        List<Integer> histogram = new ArrayList<>(m);

//        initialize list to zero
        for (int i = 0; i < m; i++) {
            histogram.add(0);
        }

        int block_index;
        int block_width = image.getWidth() / num_block_cols;
        int block_height = image.getHeight() / num_block_rows;
        int x_comp;
        int y_comp;
        int histogram_offset;

        for (int block_row = 0; block_row < num_block_rows; block_row++) {
            y_comp = block_row * block_height;
            for (int block_col = 0; block_col < num_block_cols; block_col++) {
                block_index = block_row * num_block_cols + block_col;
                x_comp = block_col * block_width;
                histogram_offset = block_index * num_bins * num_histograms; // each block has (num_bins * num_histograms) elements
                for (int y = 0; y < block_height; y++) {
                    for (int x = 0; x < block_width; x++) {
                        Color color = new Color(image.getRGB(x + x_comp, y + y_comp));
                        int red = color.getRed();
                        int green = color.getGreen();
                        int blue = color.getBlue();

                        int red_idx = (int)(red * (double)(num_bins / 256) + histogram_offset);
                        int green_idx = (int)(green * (double)(num_bins / 256) + num_bins + histogram_offset);
                        int blue_idx = (int)(blue * (double)(num_bins / 256)+ (2 * num_bins) + histogram_offset);

                        histogram.set(red_idx, histogram.get(red_idx) + 1);
                        histogram.set(green_idx, histogram.get(green_idx) + 1);
                        histogram.set(blue_idx, histogram.get(blue_idx) + 1);

                        // luminance equation is from the internet -- supposedly is the perceptual luminance given RGB
                        if (num_histograms == 4) {
                            int luminance = (int)(0.2126*red) + (int)(0.7152*green) + (int)(0.0722*blue);
                            int lum_idx = (int)(luminance * (double)(num_bins / 256) + (3 * num_bins) + histogram_offset);

                            histogram.set(lum_idx, histogram.get(lum_idx) + 1);
                        }
                    }
                }
            }
        }

        return histogram;
    }

    private static double computeCorrelation(int frameIdx1, int frameIdx2) {
        List<Integer> histogram1 = computeHistogram(frames.get(frameIdx1));
        List<Integer> histogram2 = computeHistogram(frames.get(frameIdx2));

        List<Double> norm1 = normalize_histogram(histogram1);
        List<Double> norm2 = normalize_histogram(histogram2);

        double numerator = innerProduct(norm1, norm2);
        double denominator = magnitude(norm1) * magnitude(norm2);

        return numerator/denominator;
    }

    private static double computeCorrelation(double[] vector1, double[] vector2) {
        double[] norm_vector1 = normalize_vector(vector1);
        double[] norm_vector2 = normalize_vector(vector2);

        double numerator = innerProduct(norm_vector1, norm_vector2);
        double denominator = magnitude(norm_vector1) * magnitude(norm_vector2);

//        double numerator = innerProduct(vector1, vector2);
//        double denominator = magnitude(vector1) * magnitude(vector2);

        return numerator/denominator;
    }

    private static List<Double> normalize_histogram(List<Integer> histogram) {
        final int m = histogram.size();
        List<Double> normalized_histogram = new ArrayList<>(m);
        int average = 0;
//        initialize list to zero and calculate average value
        for (Integer integer : histogram) {
            normalized_histogram.add(0.0);
            average += integer;
        }
        average /= m;

        for (int i = 0; i < m; i++) {
            normalized_histogram.set(i, (double) (histogram.get(i) - average));
        }

        return normalized_histogram;
    }

    private static double[] normalize_vector(double[] vector) {
        final int m = vector.length;
        double[] norm_vector = new double[m];
        int average = 0;
//        calculate average value
        for (Double element : vector) {
            average += element;
        }
        average /= m;

        for (int i = 0; i < m; i++) {
            norm_vector[i] = vector[i] - average;
        }

        return norm_vector;
    }



    private static double innerProduct(List<Double> vector1, List<Double> vector2) {
        final int m = vector1.size();
        if (m != vector2.size()) {
            return 0;
        }

//        List<Integer> norm_1 = normalize_histogram(histogram1);
//        List<Integer> norm_2 = normalize_histogram(histogram2);

        double sum = 0;

        for (int i = 0; i < m; i++) {
            sum += vector1.get(i) * vector2.get(i);
        }

        return sum;
    }

    private static double innerProduct(double[] vector1, double[] vector2) {
        final int m = vector1.length;
        if (m < vector2.length) {
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < m; i++) {
            sum += vector1[i] * vector2[i];
        }
        return sum;
    }

    private static double magnitude(List<Double> vector) {
        double sum = 0;

        for (Double element : vector) {
            sum += (element * element);
        }

        return Math.sqrt(sum);

    }

    private static double magnitude(double[] vector) {
        double sum = 0;

        for (Double element : vector) {
            sum += (element * element);
        }

        return Math.sqrt(sum);
    }
}

