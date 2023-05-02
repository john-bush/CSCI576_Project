import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.awt.image.BufferedImage;
import javax.sound.sampled.*;
import javax.swing.*;
import java.util.*;
import javax.swing.Timer;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;

public class VideoPlayer {
    public static JFrame frame = new JFrame("Video Display");
    public static JLabel label = new JLabel();
    public static JInternalFrame internalFrame = new JInternalFrame();
    public static JPanel panel = new JPanel();
    public static JPanel shotsPanel = new JPanel();
    public static List<BufferedImage> frames = new ArrayList<>();
    public static int currFrame = 0;
    public static File file = new File("InputVideo.rgb"); // name of the RGB video file
    public static int width = 480; // width of the video frames
    public static int height = 270; // height of the video frames
    public static int fps = 30; // frames per second of the video
    public static int numFrames = 8682; // number of frames in the video
    // this is pulled from shotTimeCodes.txt
    public static List<Integer> shotFrames = new ArrayList<Integer>(Arrays.asList(161, 251, 420, 508, 896, 1080, 1131, 1179, 1352, 1844,
    1958, 2332, 2459, 2583, 2716, 3148, 3244, 3263, 3284, 3303, 3546, 3620, 3729, 3770, 3809, 3848, 3879, 3990, 4023, 4052, 4081, 4129, 4232, 4346, 4492, 4724, 4844, 5329, 5599, 5754, 5952, 6140, 6303, 6857, 6969, 7048, 7458, 7591));
    public static List<JButton> shotButtons = new ArrayList<>();

    public static void main(String[] args) {

        AudioPlayer audioPlayer;
        try {
             audioPlayer = new AudioPlayer();
        } catch( javax.sound.sampled.UnsupportedAudioFileException |
                java.io.IOException |
                javax.sound.sampled.LineUnavailableException e) {
            System.out.println("Error with audio: " + e.getMessage());
            return;
        }

        // Set the layout of the JFrame to BorderLayout
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
        shotsPanel.setLayout(new BoxLayout(shotsPanel, BoxLayout.Y_AXIS));
        shotsPanel.setPreferredSize(new Dimension(width-50, height*2));
        JScrollPane scrollPane = new JScrollPane(shotsPanel);
        frame.add(scrollPane, BorderLayout.WEST);

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

        // Play Button
        JButton play = new JButton("Play");
        Timer playTimer = new Timer(1000/fps, null);
        play.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                playTimer.stop();
                playTimer.start();
                audioPlayer.play();
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

        // Pause Button
        JButton pause = new JButton("Pause");
        ActionListener pauseListener = new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                audioPlayer.pause();
                playTimer.stop();
            }
        };
        pause.addActionListener(pauseListener);
        panel.add(pause);

        // Shot Buttons
        for(int i = 0; i < shotFrames.size(); i++) {
            final int final_i = i;
            shotButtons.add(new JButton("Shot " + (i+1)));
            shotButtons.get(i).addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    playTimer.stop();
                    currFrame = shotFrames.get(final_i);
                    System.out.println(shotFrames.get(final_i));
                    playTimer.stop();
                    playTimer.start();

                    try {
                        double c = (double)final_i / (double)shotFrames.size();
                        long micro_time = (long)(c * (audioPlayer.clip.getMicrosecondLength()));
                        audioPlayer.jump(micro_time);
                    } catch (UnsupportedAudioFileException | IOException | LineUnavailableException err) {
                        System.out.println("Error with Audio: " + err.getMessage());
                    }


                }
            });
            shotsPanel.add(shotButtons.get(i));
        }
        panel.setBackground(Color.WHITE);
        frame.setSize(width*2, height*2);
        frame.setVisible(true);

    }

    public static void play() {
        //System.out.println(currFrame);
        label.setIcon(new ImageIcon(frames.get(currFrame)));
        internalFrame.revalidate();
        internalFrame.repaint();
        frame.validate();
        frame.repaint();
        currFrame++;

    }
}
