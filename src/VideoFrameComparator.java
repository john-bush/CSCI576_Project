import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import javax.imageio.ImageIO;

public class VideoFrameComparator {
    private static final int FRAME_RATE = 30;
    private static final int THRESHOLD = 30;
    private static final int NUM_BINS = 256;
    
    static double meanPixelIntensity = 0;

    public static void main(String[] args) {
        File file = new File("InputVideo.rgb"); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video

        double avgHistogramDist = 0;
        double avgPixelIntensityDist = 0;

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            int[] prevHistogram = new int[NUM_BINS * 3];
            double[][] prevPixelIntensity = new double[height][width];

            for (int i = 0; i < numFrames-1; i++) {
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
                int[] currHistogram = computeHistogram(image);
                double histogramDistance = computeHistogramDistance(prevHistogram, currHistogram);
                prevHistogram = currHistogram;
                avgHistogramDist += histogramDistance;
                if(histogramDistance > 9959.977089812222) {
                    System.out.println("Frame " + i + " and frame " + (i+1) + " are different by " + histogramDistance);
                }

                // Compute pixel intensities
                double[][] currPixelIntensity = computePixelIntensity(image);
                double pixelIntensityDistance = computePixelIntensityDistance(prevPixelIntensity, currPixelIntensity);
                prevPixelIntensity = currPixelIntensity;
                avgPixelIntensityDist += pixelIntensityDistance;
                if(pixelIntensityDistance > 76.07846304261982) {
                    System.out.println("Frame " + i + " and frame " + (i+1) + " are different by " + pixelIntensityDistance);
                }
                // label.setIcon(new ImageIcon(image));
                // frame.validate();
                // frame.repaint();
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

        System.out.println("MEAN: " + meanPixelIntensity);
        System.out.println((avgHistogramDist / numFrames));
        System.out.println((avgPixelIntensityDist / numFrames));
    }

    private static int[] computeHistogram(BufferedImage image) {
        // Code to compute a color histogram for the given image
        int[] histogram = new int[NUM_BINS * 3];

        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                // Get the RGB values of the pixel at (x, y)
                Color pixel = new Color(image.getRGB(x, y));
                int r = pixel.getRed();
                int g = pixel.getGreen();
                int b = pixel.getBlue();

                // Increase the bin count for each color channel
                histogram[r]++;
                histogram[NUM_BINS + g]++;
                histogram[NUM_BINS * 2 + b]++;
            }
        }
        // Return histogram as an array of integers
        return histogram;
    }

    private static double[][] computePixelIntensity(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] pixelIntensity = new double[height][width];
        int sumIntensity = 0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                // Get the RGB values of the pixel at (x, y)
                Color pixel = new Color(image.getRGB(x, y));
                int r = pixel.getRed();
                int g = pixel.getGreen();
                int b = pixel.getBlue();

                // Compute the pixel intensity
                double intensity = 0.299 * r + 0.587 * g + 0.114 * b;
                sumIntensity += ((r + g + b) / 3);
                // the order of y and x in pixelIntensity[y][x] is intentional, as it reflects the row-major order of a 2D array in Java.
                pixelIntensity[y][x] = intensity;
            }
        }
        meanPixelIntensity += (double) sumIntensity / (double) (width * height);
        return pixelIntensity;
    }

    private static double computeHistogramDistance(int[] histogram1, int[] histogram2) {
        // Code to compute the distance between two histograms
        double distance = 0.0;
        for (int i = 0; i < histogram1.length; i++) {
            // Compute the squared difference between the two histogram bins
            double diff = histogram1[i] - histogram2[i];
            distance += diff * diff;
        }
        // Normalize the distance by the number of bins
        distance /= (double) histogram1.length;
        return distance;
    }

    private static double computePixelIntensityDistance(double[][] pixelIntensity1, double[][] pixelIntensity2) {
        // Code to compute the distance between two sets of pixel intensities
        int height = pixelIntensity1.length;
        int width = pixelIntensity1[0].length;

        double distance = 0.0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double intensity1 = pixelIntensity1[y][x];
                double intensity2 = pixelIntensity2[y][x];

                // Compute the squared difference between the pixel intensities
                double diff = intensity1 - intensity2;
                distance += diff * diff;
            }
        }

        // Normalize the distance by the number of pixels
        distance /= (double) (height * width);

        // Return the distance as a double value
        return distance;
    }
}
