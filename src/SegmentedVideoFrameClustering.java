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

public class SegmentedVideoFrameClustering {
    private static final DecimalFormat df = new DecimalFormat("0.00");

    private static List<BufferedImage> frames;
    private static final int segment_length = 25; // number of frames in segment

    public static void main(String[] args) {
        File file = new File("InputVideo.rgb"); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video
        double correlation_threshold = 0.9; // TODO: need to tune this parameter

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
            for (int i = 0; i < numFrames - segment_length; i+=segment_length) {
                // compute correlation between first and last frames of segment
                double correlation = computeCorrelation(i, i+segment_length);

                if (correlation < correlation_threshold) { // segment is dynamic
                    // temporary placement of cut frame in middle of segment
                    int cut_frame = findCutFrame(i);

                    // Start a new group
                    int minutes = ((cut_frame/30) % 3600) / 60;
                    double seconds = (cut_frame/30.0) % 60;
                    System.out.println(minutes + " min : " + df.format(seconds) + " secs");
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
    static final int num_block_rows = 3;
    static final int num_block_cols = 3;
    static final int num_histograms = 4;
    static final int num_bins = 256;

    private static List<Integer> computeHistogram(BufferedImage image) {
        final int n = num_block_rows * num_block_cols * num_histograms * num_bins;
        List<Integer> histogram = new ArrayList<>(n);

//        initialize list to zero
        for (int i = 0; i < n; i++) {
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
                        int luminance = (int)(0.2126*red) + (int)(0.7152*green) + (int)(0.0722*blue);

                        int red_idx = red + histogram_offset;
                        int green_idx = green + num_bins + histogram_offset;
                        int blue_idx = blue + (2 * num_bins) + histogram_offset;
                        int lum_idx = luminance + (3 * num_bins) + histogram_offset;

                        histogram.set(red_idx, histogram.get(red_idx) + 1);
                        histogram.set(green_idx, histogram.get(green_idx) + 1);
                        histogram.set(blue_idx, histogram.get(blue_idx) + 1);
                        histogram.set(lum_idx, histogram.get(lum_idx) + 1);
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

    private static List<Double> normalize_histogram(List<Integer> histogram) {
        final int n = histogram.size();
        List<Double> normalized_histogram = new ArrayList<>(n);
        int average = 0;
//        initialize list to zero and calculate average value
        for (Integer integer : histogram) {
            normalized_histogram.add(0.0);
            average += integer;
        }
        average /= n;

        for (int i = 0; i < n; i++) {
            normalized_histogram.set(i, (double) (histogram.get(i) - average));
        }

        return normalized_histogram;
    }

    private static double innerProduct(List<Double> vector1, List<Double> vector2) {
        final int n = vector1.size();
        if (n != vector2.size()) {
            return 0;
        }

//        List<Integer> norm_1 = normalize_histogram(histogram1);
//        List<Integer> norm_2 = normalize_histogram(histogram2);

        double sum = 0;

        for (int i = 0; i < n; i++) {
            sum += vector1.get(i) * vector2.get(i);
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
}

