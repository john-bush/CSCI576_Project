import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import java.util.*;
import javax.swing.Timer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;

public class VideoPlayer {
    public static JFrame frame = new JFrame("Video Display");
    public static JLabel label = new JLabel();
    public static JInternalFrame internalFrame = new JInternalFrame();
    public static JPanel panel = new JPanel();
    public static List<BufferedImage> frames = new ArrayList<>();
    public static int currFrame = 0;
    public static File file = new File("InputVideo.rgb"); // name of the RGB video file
    public static int width = 480; // width of the video frames
    public static int height = 270; // height of the video frames
    public static int fps = 30; // frames per second of the video
    public static int numFrames = 8682; // number of frames in the video
    public static Timer timer = new Timer(0, null);

    public static void main(String[] args) {

        // Set the layout of the JFrame to BorderLayout
        //frame.setLayout(new BorderLayout());
        frame.setSize(new Dimension(width*2, height*2));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        label.setPreferredSize(new Dimension(width, height));

        // Create a JInternalFrame and add it to the JFrame
        frame.add(internalFrame, BorderLayout.EAST);

        // Set the properties of the JInternalFrame
        internalFrame.setSize(new Dimension(width, height));
        internalFrame.setVisible(true);
        internalFrame.add(label);

        // Create a JPanel and add it to the JFrame
        frame.add(panel, BorderLayout.SOUTH);
        //frame.setSize(width*2, height*2);
        //frame.setVisible(true);

        // read the video file and save each frame
        try {
            RandomAccessFile raf = new RandomAccessFile(file, "r");
            FileChannel channel = raf.getChannel();
            ByteBuffer buffer = ByteBuffer.allocate(width * height * 3);
            for (int i = 0; i < numFrames; i++) {
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
                frames.add(image);
            }
            channel.close();
            raf.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Create JButtons and add them to the JPanel
        JButton play = new JButton("Play");
        // Create an ActionListener+Timer
        Timer playTimer = new Timer(1000/fps, null);
        play.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                playTimer.stop();
                playTimer.start();
            }
        });
        playTimer.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if(currFrame < numFrames) {
                    play();
                }
            }
        });
        panel.add(play);

        JButton pause = new JButton("Pause");
        // Create an ActionListener
        ActionListener pauseListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                playTimer.stop();
                System.out.println("pause");
            }
        };
        pause.addActionListener(pauseListener);
        panel.add(pause);

        // Set the properties of the JPanel
        panel.setBackground(Color.WHITE);
        // Set the properties of the JFrame
        frame.setSize(width*2, height*2);
        frame.setVisible(true);
    }

    public static void play() {
        System.out.println(currFrame);
        label.setIcon(new ImageIcon(frames.get(currFrame)));
        internalFrame.revalidate();
        internalFrame.repaint();
        frame.validate();
        frame.repaint();
        currFrame++;
    }
}
