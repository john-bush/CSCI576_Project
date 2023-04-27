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
    public static void main(String[] args) {
        File file = new File("InputVideo.rgb"); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video
        int threshold = 10000;

        final int segment_length = 25; // number of frames in segment

        List<List<Integer>> currentGroup = new ArrayList<>();

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);

            for (int i = 0; i < numFrames-1; i+=segment_length) {
                buffer.clear();
                channel.read(buffer);
                buffer.rewind();
                BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int r = buffer.get() & 0xff;
                        int g = buffer.get() & 0xff;
                        int b = buffer.get() & 0xff;
                        int rgb = (r << 16) | (g << 8) | b;
                        image.setRGB(x, y, rgb);
                    }
                }
                // Compute histogram
                List<Integer> histogram = computeHistogram(image);
                // Add the histogram to the current group
                if(currentGroup.isEmpty()) {
                    currentGroup.add(histogram);
                }
                else {
                    List<Integer> lastHistogram = currentGroup.get(currentGroup.size() - 1);
                    double correlation = computeCorrelation(histogram, lastHistogram);
                    if (correlation > threshold) { // segment is dynamic
//                        TODO: need to find the actual cut frame within the segment

                        int cut_frame = i - (segment_length / 2);

                        // Start a new group
                        int minutes = ((cut_frame/30) % 3600) / 60;
                        double seconds = (cut_frame/30.0) % 60;
                        System.out.println(minutes + " min : " + df.format(seconds) + " secs");
                        //System.out.println(distance/10000);
                        currentGroup.clear();
                    } else { // segment is static

                    }
                    currentGroup.add(histogram);
                }

                try {
                    Thread.sleep(1000 / fps);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            channel.close();
            raf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Helper method to calculate the Euclidean distance between two histograms
    private static long computeHistogramDistance(List<Integer> histogram1, List<Integer> histogram2) {
        long distance = 0;
        for (int i = 0; i < histogram1.size(); i++) {
            long diff = histogram1.get(i) - histogram2.get(i);
            distance += diff * diff;
        }
        return distance;
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

    private static double computeCorrelation(List<Integer> vector1, List<Integer> vector2) {
        List<Double> norm1 = normalize_histogram(vector1);
        List<Double> norm2 = normalize_histogram(vector2);

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

