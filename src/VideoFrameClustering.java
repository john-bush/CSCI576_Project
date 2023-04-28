import java.awt.Color;
import java.awt.Dimension;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;
import java.text.DecimalFormat;

public class VideoFrameClustering {
    private static final DecimalFormat df = new DecimalFormat("0.00");
    public static void main(String[] args) {
        File file = new File("InputVideo.rgb"); // name of the RGB video file
        int width = 480; // width of the video frames
        int height = 270; // height of the video frames
        int fps = 30; // frames per second of the video
        int numFrames = 8682; // number of frames in the video
        int threshold = 11000;

        List<List<Integer>> currentGroup = new ArrayList<>();

        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            int lastShot = 0;

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
                List<Integer> histogram = computeHistogram(image);
                // Add the histogram to the current group
                if(currentGroup.isEmpty()) {
                    currentGroup.add(histogram);
                } 
                else {
                    List<Integer> lastHistogram = currentGroup.get(currentGroup.size() - 1);
                    long distance = computeHistogramDistance(histogram, lastHistogram);
                    if (distance/10000 > threshold) {
                        if(i - lastShot >= fps) {
                            // Start a new group
                            int minutes = ((i/30) % 3600) / 60;
                            double seconds = (i/30.0) % 60;
                            System.out.println(minutes + " min : " + df.format(seconds) + " secs");
                            //System.out.println(distance/10000);
                            lastShot = i;
                            currentGroup.clear();
                        }
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

    private static List<Integer> computeHistogram(BufferedImage image) {
        List<Integer> histogram = new ArrayList<>(256 * 3);
        for (int i = 0; i < 256; i++) {
            histogram.add(0);
            histogram.add(0);
            histogram.add(0);
        }
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                Color color = new Color(image.getRGB(x, y));
                int red = color.getRed();
                int green = color.getGreen();
                int blue = color.getBlue();
                //System.out.println(red + " " +  green + " " + blue);
                histogram.set(red, histogram.get(red) + 1);
                histogram.set(green + 256, histogram.get(green + 256) + 1);
                histogram.set(blue + 512, histogram.get(blue + 512) + 1);
            }
        }
        return histogram;
    }
}
