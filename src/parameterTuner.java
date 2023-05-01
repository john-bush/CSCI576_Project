import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class parameterTuner {


    public static void main(String[] args) {
        String videoName = "Ready_Player_One";
        double bestF1 = 0.0;

        try {
            String outputFilename = "log/ParameterTuning.txt";
            FileWriter writer = new FileWriter(outputFilename);
            BufferedWriter bw = new BufferedWriter(writer);

            SegmentedVideoFrameClustering SBD = new SegmentedVideoFrameClustering();
            SBD.setVideoName(videoName);

            int iteration_count = 0;
            for (segment_length = segment_length_min; segment_length <= segment_length_max; segment_length += 5) { // 3
                for (block_dim = block_dim_min; block_dim <= block_dim_max; block_dim++) { // 4
                    for (epsilon = epsilon_min; epsilon <= epsilon_max; epsilon *= 10) { // 3
                        for (TC2 = TC2_min; TC2 <= TC2_max; TC2 += TC2_step) { // 6
                            System.out.print(iteration_count);

                            // set run parameters
                            SBD.setSegment_length(segment_length);
                            SBD.setNum_block_cols(block_dim);
                            SBD.setNum_block_rows(block_dim);
                            SBD.setEpsilon(epsilon);
                            SBD.setTC2(TC2);

                            SBD.performSBD();

                            double F1 = calculateF1(videoName);

                            String out = ("   TC2:" + TC2 + ", e:" + epsilon + ", B:" + block_dim + ", S:" + segment_length + "\n");
                            bw.write(out);
                            System.out.print(out);
                            if (F1 > bestF1) {
                                bestF1 = F1;
                                best_epsilon = epsilon;
                                best_block_dim = block_dim;
                                best_segment_length = segment_length;
                                best_TC2 = TC2;
                            }

                            System.gc(); // garbage collection


                            iteration_count++;
                        }
                    }
                }
            }

            bw.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Execution Complete.");
        System.out.println("Best parameters: F1=" + bestF1 + best_segment_length + ", block_dim=" + best_block_dim + ", epsilon=" + epsilon + ", TC2=" + TC2);
    }

    private static double calculateF1 (String videoName) {
        String videoLogName = "lib/"+videoName+"_rgb/shotTimeCodes.txt";
        String testLogName = "log/" + videoName + "_B"+block_dim+"x"+block_dim+"_Thresh1_"+TC1+"_Thresh2_" + TC2 + "_E_" + epsilon + "_Seg_" + segment_length + "_FStep_" + frame_step;
        List<Integer> test = readLog(testLogName);
        List<Integer> real = readLog(videoLogName);

        int[] N = calculateN(test, real);

        double Nc = N[0];
        double Nf = N[1];
        double Nm = N[2];

        double P = Nc / (Nc + Nf);
        double R = Nc / (Nc + Nm);
        double F1 = (2 * P * R) / (P + R);

        System.out.print(":    F1=" + F1 + ",   Nc=" + Nc + ",  Nf=" + Nf + ",  Nm=" + Nm);

        return F1;
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

    private static List<Integer> readLog(String filename) {
        List<Integer> out = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            reader.readLine(); // throw away header line
            String line = reader.readLine(); // get first integer line
            String frameNumString = "";
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

    static final int block_dim_min = 3;
    static final int block_dim_max = 5;
    private static final int segment_length_min = 20; // number of frames in segment
    private static final int segment_length_max = 25; // number of frames in segment
    private static final int frame_step_min = 3; // frame step used for cut verification
    private static final int frame_step_max = 6; // frame step used for cut verification

    //    THRESHOLDS AND TUNING PARAMETERS
    static final double epsilon_min = 0.0001;
    static final double epsilon_max = 0.01;
    static final double TC1_min = 0.5;
    static final double TC1_max = 0.8;
    static final double TC1_step = 0.1;
    static final double TC2_min = 0.89;
    static final double TC2_max = 0.96;
    static final double TC2_step = 0.0200;

    private static int best_block_dim = 3;
    private static int best_segment_length = 20;
    private static int best_frame_step = 3;
    private static double best_TC1 = 0.74;
    private static double best_TC2 = 0.85;
    private static double best_epsilon = 0.0001;


    private static int block_dim = 3;
    private static int num_histograms = 4;
    private static int num_bins = 256;
    private static int segment_length = 25; // number of frames in segment
    private static int frame_step = 4; // frame step used for cut verification

    //    THRESHOLDS AND TUNING PARAMETERS
    private static double epsilon = 0.0001;
    private static double TC1 = 0.75;
    private static double TC2 = 0.90;
}
