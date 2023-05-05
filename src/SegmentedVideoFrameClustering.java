import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.*;
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
import org.apache.commons.math4.legacy.linear.SparseRealMatrix;

public class SegmentedVideoFrameClustering {
    private static final DecimalFormat df = new DecimalFormat("000.00");

    public SegmentedVideoFrameClustering() {
    }

    public SegmentedVideoFrameClustering(String VideoName) {
        videoName = VideoName;
    }

    public void setFrames(List<BufferedImage> frames) {
        SegmentedVideoFrameClustering.frames = frames;
    }

    public TreeMap<Integer, List<Integer>> getShotFrames() {
        return shotFrames;
    }

    public static TreeMap<Integer, List<Integer>> shotFrames = new TreeMap<>();
    public static List<Integer> shotBoundaries = new ArrayList<>(); // contains first frame idx of each shot

    public List<BufferedImage> getFrames() {
        return frames;
    }

    private static List<BufferedImage> frames = new ArrayList<>();

    private static int num_block_rows = 3;
    private static int num_block_cols = 3;
    private static int num_histograms = 3;
    private static final int num_bins = 256;
    private static int segment_length = 25; // number of frames in segment
    private static int frame_step = 4; // frame step used for cut verification

//    THRESHOLDS AND TUNING PARAMETERS
    private static double epsilon = 0.001;
    private static double TC1 = 0.75;
    private static double TC2 = 0.92;
    static final double correlation_threshold = 0.96;
    static final int DEBUG = -1; // to show correlation differences
    static final boolean PRINT_TIME = false;

    private static String videoName = "Ready_Player_One";

    private boolean framesRead = false;

    public int getNumFrames() {
        return numFrames;
    }

    private int numFrames = 8682; // number of frames in the video

    private static int curr_segment_frame;

    // Main method
    public static void main(String[] args) {
        SegmentedVideoFrameClustering SBD = new SegmentedVideoFrameClustering(); // Create an object of Main
        SBD.run();
    }

    public void run() {
        System.out.println("Performing SBD...");
        performSBD(); // Call the public method on the object
        System.out.println("Grouping shots into scenes...");
        List<Integer> sceneBoundaries = getScenes();
        System.out.println("Assembled Scene Heirarchy:");
        assembleScenes(sceneBoundaries);
    }

    public void performSBD() {
//        String videoName = "The_Great_Gatsby";
//        String videoName = "Ready_Player_One";


        String outputFilename = "log/" + videoName + "_B"+num_block_rows+"x"+num_block_cols+"_Thresh1_"+TC1+"_Thresh2_" + TC2 + "_E_" + epsilon + "_Seg_" + segment_length + "_FStep_" + frame_step;
        String header = "Blocks:"+num_block_rows+"x"+num_block_cols+", Thresholds: "+TC1+", " + TC2 + ", Epsilon: " + epsilon;
        shotBoundaries.add(0);
        try {

            FileWriter writer = new FileWriter(outputFilename);
            BufferedWriter bw = new BufferedWriter(writer);

//            System.out.println(header);
            bw.write(header + "\n");

            if (!framesRead) {
                frames = readVideoFile(videoName);
                framesRead = true;
            }

            // calculate segment beginning-end frame correlations
            for (curr_segment_frame = 0; curr_segment_frame < numFrames - segment_length - 1; curr_segment_frame+=(segment_length - 1)) {
                // compute correlation between first and last frames of segment
                double correlation = computeCorrelation(curr_segment_frame, curr_segment_frame + segment_length - 1);

                if (DEBUG > 2) {
                    System.out.println("Segment Correlation: " + correlation);
                }

                if (correlation <= correlation_threshold) { // segment is dynamic
//      perform cut verification as covered in the paper
                    int rel_frame = cutTransitionID_Incremental(curr_segment_frame);
//                    int rel_frame = cutTransitionID(i);
                    if (rel_frame != -1) {
                        if (DEBUG == 2) {
                            System.out.print("Segment Frame: " + curr_segment_frame + ", Local Frame: " + rel_frame + ", Shot: ");
                        }
                        int cut_frame = curr_segment_frame + rel_frame; // first frame of next scene
                        shotBoundaries.add(cut_frame);
                        if (PRINT_TIME) {
                            // Start a new group
                            int minutes = ((cut_frame / 30) % 3600) / 60;
                            double seconds = (cut_frame / 30.0) % 60;
                            String timestamp = minutes + " min : " + df.format(seconds) + " secs";
                            System.out.println(timestamp);
                            bw.write(timestamp + "\n");
                        } else {
                            if (DEBUG >= 0) {
                                System.out.println(cut_frame);
                            }
                            bw.write(cut_frame + "\n");
                        }

                    }
                    //System.out.println(distance/10000);
                } // else { // segment is static, do nothing
            }
            shotBoundaries.add(numFrames);

            double[] F1 = calculateF1(videoName);
            bw.write("F1=" + F1[0] + ",   Nc=" + F1[1] + ",  Nf=" + F1[2] + ",  Nm=" + F1[3]);
            bw.close();
            writer.close();

            System.out.println("   TC2:" + TC2 + ", e:" + epsilon + ", B:" + num_block_rows + ", S:" + segment_length + "\n");

//            try {
//                Thread.sleep(1000 / fps);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }

        } catch (IOException e) {
            e.printStackTrace();
        }

//        frames.clear();
//        System.gc();
    }

    public void assembleScenes(List<Integer> sceneBoundaries) { // frame number of the start of the last shot in the scene
        int prevSceneEnd = -1;
        for (int sceneIdx = 0; sceneIdx < sceneBoundaries.size(); sceneIdx++) {
            int sceneEnd = sceneBoundaries.get(sceneIdx); // index of last shot of scene
            List<Integer> shots = new ArrayList<>();
            int frameIdx = shotBoundaries.get(prevSceneEnd + 1);
            System.out.println("Scene " + (sceneIdx + 1) + ": " + frameIdx);
            int counter = 1;
            for (int shotIdx = prevSceneEnd + 1; shotIdx <= sceneEnd; shotIdx++) {
                frameIdx = shotBoundaries.get(shotIdx);
                shots.add(frameIdx);
                System.out.println("\tShot " + counter + ":" + "Frame " + frameIdx);
                counter++;
            }
            shotFrames.put(shots.get(0), shots);
            prevSceneEnd = sceneEnd;
        }
    }


    // scene clustering given shot boundaries https://www.ibm.com/blogs/research/2018/09/video-scene-detection/
    // things we need: from https://github.com/ivi-ru/video-scene-detection/blob/master/h_nrm.py
//          - [DONE] pairwise distance matrix between shots (what is the feature that we are calculating distance based on??)
//                              -> for now, use averaged color histogram across shot
//          - [DONE] calculate number of scenes in the video
//          - [DONE] 2D array sum (int a, int b)-> return sum of elements in a square subset of a matrix from [a->b][a->b]
//          - [DONE] get_embedded_areas_sums() function (i dunno its in the github)
//          - get_optimal_sequence_nrm() perform normalized cost function to get scene boundaries
    public double distance_sum(double[][] distanceMatrix, int a, int b) {
        double sum = 0.0;
        for (int i = a - 1; i < b; i++) {
            for (int j = a - 1; j < b; j++) {
                sum += distanceMatrix[i][j];
            }
        }
        return sum;
    }

    public int estimate_scene_count(double[][] distanceMatrix) {
        SingularValueDecomposition svd = new SingularValueDecomposition(new Array2DRowRealMatrix(distanceMatrix));
        double[] rawSingulars = svd.getSingularValues();
        double[] singularValues = new double[(int)(rawSingulars.length * 0.75)];

        for(int i = 0; i < singularValues.length; i++) {
            singularValues[i] = Math.log(rawSingulars[i]);
        }

        double[] startPoint = {0, singularValues[0]};
        double[] endPoint = {singularValues.length, singularValues[singularValues.length - 1]};
        double max_distance = 0.0;
        int elbow_point = 0;

        for (int i = 0; i < singularValues.length; i++) {
            double[] current_point = {i, singularValues[i]};
            double[] a = {startPoint[0] - endPoint[0], startPoint[1] - endPoint[1]};
            double[] b = {startPoint[0] - current_point[0], startPoint[1] - current_point[1]};
            double[] c = {endPoint[0] - startPoint[0], endPoint[1] - startPoint[1]};
            double distance = Math.abs((a[0] * b[1] - b[0] * a[1]) / magnitude(c));
            if (distance > max_distance) {
                max_distance = distance;
                elbow_point = i;
            }
        }
        return elbow_point;
    }

    private static final int shot_padding = 4;
    public double[][] pairwiseDistances() {
        int dim = shotBoundaries.size() - 1; // number of shots in whole video
        double[][] distances = new double[dim][dim];
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;
        double[][] averageHistograms = new double[m][dim];
        for (int i = 0; i < dim; i++) {
            int shotStart = shotBoundaries.get(i);
            int shotEnd = shotBoundaries.get(i+1) - 1;
            averageHistograms[i] = averageHistogram(shotStart + shot_padding, shotEnd - shot_padding, 3);
        }
        for (int i = 0; i < dim - 1; i++) {
            for (int j = i; j < dim - 1; j++) {
                double correlation = computeCorrelation(averageHistograms[i], averageHistograms[j]);
//                double adj_correlation = Math.pow(-4.0 * correlation + 4, 3);
                double adj_correlation = 1.0 - correlation;
                distances[i][j] = adj_correlation; // convert cosine similarity to a distance (greater is further apart)
                distances[j][i] = adj_correlation;
//                double distance = euclideanDistance(averageHistograms[i], averageHistograms[j]);
//                distances[i][j] = distance;
//                distances[j][i] = distance;
            }
        }
        return distances;
    }

    public Set<Integer> get_embedded_areas_sums(int parent_square_size, int embedded_squares_count) {
        Set<Integer> hash_Set = new HashSet<>();
        if (embedded_squares_count == 0) {
            hash_Set.add(0);
            return hash_Set;
        }
        if (embedded_squares_count == 1) {
            hash_Set.add(parent_square_size * parent_square_size);
            return hash_Set;
        }
        if ((parent_square_size / (double)embedded_squares_count) < 1) {
            return hash_Set;
        }

        for (int i = 0; i < (int)Math.round(parent_square_size / 2.); i++) {
            int area = (i + 1) * (i + 1);
            Set<Integer> recursiveSet = get_embedded_areas_sums(parent_square_size - i - 1, embedded_squares_count - 1);
            for (Integer element :  recursiveSet) {
                hash_Set.add(area + element);
            }
        }
        return hash_Set;
    }

    public List<Integer> getScenes() {
        double[][] D = pairwiseDistances();
        System.out.println();
        for (double[] doubles : D) {
            System.out.print("[");
            for (int y = 0; y < D.length; y++) {
                System.out.print(df.format(doubles[y]) + ", ");
            }
            System.out.println("],");
        }
        int K = (int)(estimate_scene_count(D) * 1.50);
        System.out.println("Estimate Scene Count: " + K);
        int N = D.length;
        HashMap<List<Integer>, Double> C = new HashMap<>();
        HashMap<List<Integer>, Integer> J = new HashMap<>();
        HashMap<List<Integer>, Integer> P = new HashMap<>();

        int k = 1;
        for (int n = 1; n < N+1; n++) {
            int _area = (N - n + 1) * (N - n + 1);
            var sums = get_embedded_areas_sums(n - 1, K - k);
            for (Integer p : sums) {
                double _dist = distance_sum(D, n, N);
                J.put(Arrays.asList(n, k, p), N);
                P.put(Arrays.asList(n, k, p), _area);
                C.put(Arrays.asList(n, k, p), (_dist / (p + _area)));
            }
        }

        for (k = 2; k < K + 1; k++) {
            for (int n = 1; n < N; n++) {
                if ((N - n + 1) < k) {
                    continue;
                }
                for (Integer p : get_embedded_areas_sums(n - 1, K - k)) {
                    double min_C = Double.MAX_VALUE;
                    int min_i = Integer.MAX_VALUE;
//                    c = G + C.get((i + 1, k - 1, p + cur_area), 0)
//
//                    if c < min_C:
//                        min_C = c
//                        min_i = i

                    for (int i = n; i < N; i++) {
                        if ((N - i) < (k - 1)) {
                            continue;
                        }
                        int curr_area = (i - n + 1) * (i - n + 1);
                        Integer next_area = P.get(Arrays.asList(i + 1, k - 1, p + curr_area));
                        if (next_area == null) {
                            next_area = 0;
                        }
                        double G = distance_sum(D, n, i) / (p + curr_area + next_area);
                        Double c_ = C.get(Arrays.asList(i + 1, k - 1, p + curr_area));
                        if (c_ == null) {
                            c_ = 0.;
                        }
                        double c = G + c_;
                        if (c < min_C) {
                            min_C = c;
                            min_i = i;
                        }
                    }
                    C.put(Arrays.asList(n, k, p), min_C);
                    J.put(Arrays.asList(n, k, p), min_i);

                    int m = (min_i - n + 1) * (min_i - n + 1);
                    Integer tempVal = P.get(Arrays.asList(min_i + 1, k - 1, p + m));
                    if (tempVal == null) {
                        tempVal = 0;
                    }
                    P.put(Arrays.asList(n, k, p), m + tempVal);
                }
            }
        }
        List<Integer> boundaries = new ArrayList<>(); // will contain index of last shot in scene
        boundaries.add(0);
        int P_tot = 0;
        for (int i = 1; i < K + 1; i++) {
            int boundary = J.get(Arrays.asList(boundaries.get(boundaries.size() - 1) + 1, K - i + 1, P_tot));
//            t.append(J[(t[-1] + 1, K - i + 1, P_tot)])
            boundaries.add(boundary);
            int val = (boundaries.get(boundaries.size() - 1) - boundaries.get(boundaries.size() - 2));
            P_tot += (val * val);
        }
        boundaries.remove(0);
        int index = 0;
        for (Integer element : boundaries) {
            boundaries.set(index++, element - 1);
        }
        return boundaries;
    }
    /**
     *
     *     t = [0]
     *     P_tot = 0
     *     for i in range(1, K + 1):
     *         t.append(J[(t[-1] + 1, K - i + 1, P_tot)])
     *         P_tot += (t[-1] - t[-2]) ** 2
     *     return np.array(t[1:]) - 1
    */
    public double[] averageHistogram(int startFrame, int endFrame, int step) {
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;
        double[] avg = new double[m];
        if (startFrame >= endFrame) {
            startFrame--;
            endFrame++;
        }
        int count = 0;
        for (int i = startFrame; i < endFrame; i+=step) {
            if (i > endFrame) {
                i = endFrame;
            }
            count++;
            int[] currFrame = computeHistogramVector(frames.get(i), false);
            for (int idx = 0; idx < currFrame.length; idx++) {
                avg[idx] = avg[idx] + currFrame[idx];
            }
        }

        for (int i = 0; i < avg.length; i++) {
            avg[i] = avg[i] / (double)count;
        }

        return avg;
    }

    public List<BufferedImage> readVideoFile(String videoName) {
        List<BufferedImage> frameList = new ArrayList<>();
        String videoPathName = "lib/"+videoName+"_rgb/InputVideo.rgb";
        File file = new File(videoPathName); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            long fileSize = channel.size(); // get the size of the file
            int frameCounter = 0;
            boolean notSegmentMultiple = false;
            while (channel.position() < fileSize || notSegmentMultiple) {
                frameCounter++;
                buffer.clear();
                channel.read(buffer);
                buffer.rewind();
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                frameList.add(image);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        image.setRGB(x, y, rgb);
                    }
                }
                notSegmentMultiple = frameCounter % segment_length != 0;
            }
            numFrames = frameCounter;
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
        return frameList;
    }



    /**
     *
     * @param index - frame index of start of segment to analyze
     * @return frame number of cut transition (-1 if the segment is static)
     */
    private int cutTransitionID(int index) {
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
            int idx2 = d * frame_step + frame_step;
            if (idx2 >= matrixBeta.length) {
                idx2 = matrixBeta.length - 1;
            }
            double[] frame2 = matrixBeta[idx2];
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
                System.out.println("Returned on TC1 at frame " + index);
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
        System.out.println("TC2 Threshold Count for Segment at frame " + index + " is " + threshold_count);
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

    private int cutTransitionID_Incremental(int index) {
        double[][] matrixBeta = calculateMatrixBeta(index);
        int cut_index = -1;
        // look for distance less than TC1
        // if a distance is less than TC1 is found, calculate the distances between all pairs of frames
        // within that frame step. The frame pair with a distance less than TC1 is where the cut frame is

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
        System.out.println("TC2 Threshold Count for Segment at frame " + index + " is " + threshold_count);
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
            int last_boundary = shotBoundaries.get(shotBoundaries.size() - 1) + 15;
            if (last_boundary >= (cut_index + curr_segment_frame)) {
                return -1;
            } else {
                return cut_index;
            }
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
    private double[][] constructMatrixH(int index) {
        // length of each column vector
        final int m = num_block_rows * num_block_cols * num_histograms * num_bins;

        // construct matrix H by joining CBBH column vector of each frame
        double[][] H = new double[m][segment_length];

        // calculates each frame histogram and sets a column of H
        for (int i = 0; i < segment_length; i++) {
            int[] column = computeHistogramVector(frames.get(i + index), true);
            for (int r = 0; r < m; r++) {
                H[r][i] = column[r];
            }
        }

        return H;
    }

    private static int findK(double[][] Sigma, int r) {
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

    private double[][] calculateMatrixBeta(int index) {
        int minutes = ((index / 30) % 3600) / 60;
        double seconds = (index / 30.0) % 60;
        if (DEBUG > 0) {
            System.out.print("Processing segment starting at " + minutes + ":" + df.format(seconds) + "s");

        }
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
    }

    /**
     *  computeHistogram(BufferedImage image)
     * @param image - rgb buffered image
     * @return calculates a concatenated block histogram (CBBH) of the image.
     *          breaks the image into 3x3 blocks and calculates the R, G, B and Y
     *          histograms for each block.
     *          (each of the 9 block's histograms are all concatenated together, such as:
     *              [B1R, B1G, B1B, B1Y, B2R, B2G, B2B, B2Y, ...]
     */

    private List<Integer> computeHistogram(BufferedImage image, boolean useLuminance) {
        if (useLuminance) {
            num_histograms = 4;
        } else {
            num_histograms = 3;
        }
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

                        // luminance equation is from the internet -- supposedly is the perceptual luminance given RGB
                        if (useLuminance) {
                            int luminance = getLuminance(red, green, blue);
                            int lum_idx = (int) (luminance * (double) (num_bins / 256) + (3 * num_bins) + histogram_offset);

                            histogram.set(lum_idx, histogram.get(lum_idx) + 1);
                        } else {
                            int[] rgb = {red, green, blue};
                            int[] yiq = convertRGBtoYIQ(rgb);
                            red = yiq[0];
                            green = yiq[1];
                            blue = yiq[2];
                        }

                        int red_idx = (int)(red * (double)(num_bins / 256) + histogram_offset);
                        int green_idx = (int)(green * (double)(num_bins / 256) + num_bins + histogram_offset);
                        int blue_idx = (int)(blue * (double)(num_bins / 256)+ (2 * num_bins) + histogram_offset);

                        histogram.set(red_idx, histogram.get(red_idx) + 1);
                        histogram.set(green_idx, histogram.get(green_idx) + 1);
                        histogram.set(blue_idx, histogram.get(blue_idx) + 1);


                    }
                }
            }
        }

        return histogram;
    }

    private int[] computeHistogramVector(BufferedImage image, boolean useLuminance) {
        int num_histograms_;
        if (useLuminance) {
            num_histograms_ = 4;
        } else {
            num_histograms_ = 3;
        }
        int m = num_block_rows * num_block_cols * num_histograms_ * num_bins;

        int[] histogram = new int[m];

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
                histogram_offset = block_index * num_bins * num_histograms_; // each block has (num_bins * num_histograms) elements
                for (int y = 0; y < block_height; y++) {
                    for (int x = 0; x < block_width; x++) {
                        Color color = new Color(image.getRGB(x + x_comp, y + y_comp));
                        int red = color.getRed();
                        int green = color.getGreen();
                        int blue = color.getBlue();

                        if (useLuminance) {
                            // luminance equation is from the internet -- supposedly is the perceptual luminance given RGB
                            int luminance = getLuminance(red, green, blue);
                            int lum_idx = (int)(luminance * (double)(num_bins / 256) + (3 * num_bins) + histogram_offset);

                            histogram[lum_idx] = histogram[lum_idx] + 1;
                        }
                        else { // use YIQ
                            int[] rgb = {red, green, blue};
                            int[] yiq = convertRGBtoYIQ(rgb);
                            red = yiq[0];
                            green = yiq[1];
                            blue = yiq[2];
                        }

                        int red_idx = (int)(red * (double)(num_bins / 256) + histogram_offset);
                        int green_idx = (int)(green * (double)(num_bins / 256) + num_bins + histogram_offset);
                        int blue_idx = (int)(blue * (double)(num_bins / 256)+ (2 * num_bins) + histogram_offset);

                        histogram[red_idx] = histogram[red_idx] + 1;
                        histogram[green_idx] = histogram[green_idx] + 1;
                        histogram[blue_idx] = histogram[blue_idx] + 1;


                    }
                }
            }
        }

        return histogram;
    }

    private double computeCorrelation(int frameIdx1, int frameIdx2) {
        List<Integer> histogram1 = computeHistogram(frames.get(frameIdx1), false);
        List<Integer> histogram2 = computeHistogram(frames.get(frameIdx2), false);

        List<Double> norm1 = normalize_histogram(histogram1);
        List<Double> norm2 = normalize_histogram(histogram2);

        double numerator = innerProduct(norm1, norm2);
        double denominator = magnitude(norm1) * magnitude(norm2);

        return numerator/denominator;
    }

    private double computeCorrelation(double[] vector1, double[] vector2) {
        double[] norm_vector1 = normalize_vector(vector1);
        double[] norm_vector2 = normalize_vector(vector2);

        double numerator = innerProduct(norm_vector1, norm_vector2);
        double denominator = magnitude(norm_vector1) * magnitude(norm_vector2);

//        double numerator = innerProduct(vector1, vector2);
//        double denominator = magnitude(vector1) * magnitude(vector2);
        if (denominator == 0.) {
            return 0;
        }
        return numerator/denominator;
    }

    private List<Double> normalize_histogram(List<Integer> histogram) {
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

    private double[] normalize_vector(double[] vector) {
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

    private double euclideanDistance(double[] A, double[] B) {
        double[] A_ = normalize_vector(A);
        double[] B_ = normalize_vector(B);
        double[] temp = new double[A.length];
        for (int i = 0; i < A_.length; i++) {
            temp[i] = (A_[i] - B_[i]) * (A_[i] - B_[i]);
        }
        double sum = 0;
        for (Double element : temp) {
            sum += element;
        }
        return Math.sqrt(sum);
    }



    private double innerProduct(List<Double> vector1, List<Double> vector2) {
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

    private double innerProduct(double[] vector1, double[] vector2) {
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

    private double magnitude(List<Double> vector) {
        double sum = 0;

        for (Double element : vector) {
            sum += (element * element);
        }

        return Math.sqrt(sum);

    }

    private double magnitude(double[] vector) {
        double sum = 0;

        for (Double element : vector) {
            sum += (element * element);
        }

        return Math.sqrt(sum);
    }

    /**
     * @param r color in [0, 255]
     * @param g color in [0, 255]
     * @param b color in [0, 255]
     * @return the linear luminance value with a gamma correction within range 0-255
     */
    private int getLuminance(int r, int g, int b) {
        double rawLum = (0.2126 * sRGBtoLin(r) + 0.7152 * sRGBtoLin(g) + 0.0722 * sRGBtoLin(b));
        return (int)(rawLum * 255);
    }

    private double sRGBtoLin(int sRGBVal) {
        double linearVal = sRGBVal / 255.0;
        if ( linearVal <= 0.04045 ) {
            return linearVal / 12.92;
        } else {
            return Math.pow((( linearVal + 0.055)/1.055),2.4);
        }
    }

    public int[] convertRGBtoYIQ(int[] rgb) {
        float r = rgb[0];
        float g = rgb[1];
        float b = rgb[2];

        int[] yiq = new int[3];

        yiq[0] = (int)((0.299900f * r) + (0.587000f * g) + (0.114000f * b));
        yiq[1] = (int)((0.595716f * r) - (0.274453f * g) - (0.321264f * b));
        yiq[2] = (int)((0.211456f * r) - (0.522591f * g) + (0.311350f * b));

        return yiq;
    }

    private static double[] calculateF1 (String videoName) {
        String videoLogName = "lib/"+videoName+"_rgb/shotTimeCodes.txt";
        List<Integer> test = new ArrayList<>();
        for (Integer element : shotBoundaries) {
            test.add(element);
        }
        List<Integer> real = readLog(videoLogName);

        int[] N = calculateN(test, real);

        double Nc = N[0];
        double Nf = N[1];
        double Nm = N[2];

        double P = Nc / (Nc + Nf);
        double R = Nc / (Nc + Nm);
        double F1 = 2 * ( (P * R) / (P + R) );

        System.out.print("F1=" + F1 + ",   Nc=" + Nc + ",  Nf=" + Nf + ",  Nm=" + Nm);

        return new double[]{F1, Nc, Nf, Nm};

    }

    private static List<Integer> readLog(String filename) {
        List<Integer> out = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            reader.readLine(); // throw away header line
            String line = reader.readLine(); // get first integer line
            String frameNumString;
            int number;
            while (line != null) {
//                System.out.println(line);
                frameNumString = line.split(" ")[0];
                number = Integer.parseInt(frameNumString);
                out.add(number);
                line = reader.readLine();
            }

        } catch (IOException e) {
            System.err.format("IOException: %s%n", e);
        }

        return out;
    }


    private static final int softness = 7; // we will consider a frame +/-softness from the real timestamp as correct
    private static int[] calculateN (List<Integer> test, List<Integer> real) {
        int[] N = {0, 0, 0};
        int Nc = 0;
        List<Integer> detectedTest = new ArrayList<>();
        List<Integer> detectedReal = new ArrayList<>();
        for (Integer element : real) {
            for (int frame = element-softness; frame <= element+softness; frame++) {
                if (test.contains(frame)) {
                    detectedTest.add(frame);
                    detectedReal.add(element);
                    Nc++;
                }
            }
        }

        for (Integer element : detectedTest) {
            test.remove(element);
        }
        for (Integer element : detectedReal) {
            real.remove(element);
        }

        N[0] = Nc;
        N[1] = test.size();
        N[2] = real.size();

        return N;
    }

    public void setVideoName(String videoName) {
        SegmentedVideoFrameClustering.videoName = videoName;
    }

    public void setNum_block_rows(int num_block_rows) {
        SegmentedVideoFrameClustering.num_block_rows = num_block_rows;
    }

    public void setNum_block_cols(int num_block_cols) {
        SegmentedVideoFrameClustering.num_block_cols = num_block_cols;
    }

    public void setSegment_length(int segment_length) {
        SegmentedVideoFrameClustering.segment_length = segment_length;
    }

    public void setFrame_step(int frame_step) {
        SegmentedVideoFrameClustering.frame_step = frame_step;
    }

    public void setEpsilon(double epsilon) {
        SegmentedVideoFrameClustering.epsilon = epsilon;
    }

    public void setTC1(double TC1) {
        SegmentedVideoFrameClustering.TC1 = TC1;
    }

    public void setTC2(double TC2) {
        SegmentedVideoFrameClustering.TC2 = TC2;
    }
}

