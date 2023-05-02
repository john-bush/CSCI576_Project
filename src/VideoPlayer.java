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
import java.util.Map;
import java.util.TreeMap;

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
    public static int numFrames = 0; // number of frames in the video
    public static TreeMap<Integer, List<Integer>> shotFrames = new TreeMap<>();
    public static List<JButton> hierarchyButtons = new ArrayList<>();

    public static void main(String[] args) {
        // manually adding for testing
        shotFrames.put(161, Arrays.asList(251, 420, 508));
        shotFrames.put(896, Arrays.asList(1080, 1131, 1179, 1352, 1844));
        shotFrames.put(1958, Arrays.asList(2332, 2459, 2583, 2716, 3148, 3244, 3263));
        shotFrames.put(3284, Arrays.asList(3303, 3546, 3620, 3729, 3770, 3809, 3848, 3879, 3990, 4023, 4052, 4081));
        shotFrames.put(4129, Arrays.asList(4232, 4346, 4492, 4724, 4844, 5329, 5599, 5754, 5952, 6140, 6303, 6857, 6969, 7048, 7458, 7591));
        
        numFrames = (int)(file.length() / (352*288*(fps/8)));

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
        JScrollPane scrollPane = new JScrollPane(shotsPanel, JScrollPane.VERTICAL_SCROLLBAR_ALWAYS, 
        JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setPreferredSize(new Dimension(width-50, height*2));
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

        // Scene and Shot Buttons
        int i = 1;
        for (int sceneFrame : shotFrames.keySet()) {
            hierarchyButtons.add(new JButton("Scene " + (i)));
            hierarchyButtons.get(hierarchyButtons.size()-1).addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    playTimer.stop();
                    currFrame = sceneFrame;
                    System.out.println(sceneFrame);
                    playTimer.stop();
                    playTimer.start();
                    try {
                        double c = (double)sceneFrame / (double)shotFrames.size();
                        long micro_time = (long)(c * (audioPlayer.clip.getMicrosecondLength()));
                        audioPlayer.jump(micro_time);
                    } catch (UnsupportedAudioFileException | IOException | LineUnavailableException err) {
                        System.out.println("Error with Audio: " + err.getMessage());
                    }
                }
            });
            shotsPanel.add(hierarchyButtons.get(hierarchyButtons.size()-1));
            i++;
            int j = 1;
            for(int shotFrame : shotFrames.get(sceneFrame)) {
                hierarchyButtons.add(new JButton("Shot " + (j)));
                hierarchyButtons.get(hierarchyButtons.size()-1).addActionListener(new ActionListener() {
                    @Override
                    public void actionPerformed(ActionEvent e) {
                        playTimer.stop();
                        currFrame = shotFrame;
                        System.out.println(shotFrame);
                        playTimer.stop();
                        playTimer.start();
                        try {
                            double c = (double)shotFrame / (double)shotFrames.size();
                            long micro_time = (long)(c * (audioPlayer.clip.getMicrosecondLength()));
                            audioPlayer.jump(micro_time);
                        } catch (UnsupportedAudioFileException | IOException | LineUnavailableException err) {
                            System.out.println("Error with Audio: " + err.getMessage());
                        }
                    }
                });
                shotsPanel.add(hierarchyButtons.get(hierarchyButtons.size()-1));
                j++;
            }
        }
        panel.setBackground(Color.WHITE);
        frame.setSize(width*2, height*2);
        frame.setVisible(true);
    }

    public static void play() {
        label.setIcon(new ImageIcon(frames.get(currFrame)));
        internalFrame.revalidate();
        internalFrame.repaint();
        frame.validate();
        frame.repaint();
        currFrame++;
    }
}
