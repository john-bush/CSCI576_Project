import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.*;

public class SceneDetection {

    public static void main(String[] args) {
        String audioPath = "InputAudio.wav";

        int[] sceneDetectionTimes = detectScenes(audioPath);

//        // print scene detection times in seconds
//        for (int time : sceneDetectionTimes) {
//            System.out.println("Scene detected at " + time + " seconds");
//        }
    }

    public static int[] detectScenes(String audioPath) {
        // read audio file
        File audioFile = new File(audioPath);
        AudioInputStream audioStream = null;
        try {
            audioStream = AudioSystem.getAudioInputStream(audioFile);
        } catch (Exception e) {
            System.out.println("Error reading audio file: " + e.getMessage());
            return new int[0];
        }

        byte[] audioBytes = null;
        try {
            AudioFormat format = audioStream.getFormat();
            int numBytes = (int) (audioStream.getFrameLength() * format.getFrameSize());
            audioBytes = new byte[numBytes];
            int numBytesRead = audioStream.read(audioBytes, 0, numBytes);
            if (numBytesRead != numBytes) {
                System.out.println("Error reading audio data");
                return new int[0];
            }
        } catch (IOException e) {
            System.out.println("Error extracting audio data: " + e.getMessage());
            return new int[0];
        }
        double[] audioData = new double[audioBytes.length / 2];
        for (int i = 0; i < audioData.length; i++) {
            audioData[i] = ((short) ((audioBytes[2 * i + 1] << 8) | audioBytes[2 * i])) / 32768.0;
        }

        // set parameters for STFT
        int windowSize = 131072;
        int hopSize = 65536;

        // apply STFT to audio data based on the window and hop size
        FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
        ArrayList<double[]> spectrogramList = new ArrayList<double[]>();
        for (int i = 0; i < audioData.length - windowSize; i += hopSize) {
            double[] frame = new double[windowSize];
            for (int j = 0; j < windowSize; j++) {
                frame[j] = audioData[i + j];
            }
            Complex[] fft = transformer.transform(frame, TransformType.FORWARD);
            double[] spectrum = new double[fft.length / 2];
            for (int j = 0; j < spectrum.length; j++) {
                spectrum[j] = fft[j].abs();
            }
            spectrogramList.add(spectrum);
        }
        double[][] spectrogram = new double[spectrogramList.size()][];
        for (int i = 0; i < spectrogramList.size(); i++) {
            spectrogram[i] = spectrogramList.get(i);
        }
        double[] meanEnergy = new double[spectrogram.length];


        for (int i = 0; i < meanEnergy.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < spectrogram.length; j++) {
                sum += spectrogram[i][j];
            }
            meanEnergy[i] = sum / spectrogram.length;
        }


        double maxEnergy = Double.MIN_VALUE;
        for (int i = 0; i < meanEnergy.length; i++) {
            if (meanEnergy[i] > maxEnergy) {
                maxEnergy = meanEnergy[i];
            }
        }

        // Determine the threshold needed to find the scene changes
        double threshold = maxEnergy * 0.6;

        ArrayList<Double> sceneChangeTimesList = new ArrayList<Double>();
        for (int i = 0; i < meanEnergy.length; i++) {
            if (meanEnergy[i] > threshold) {
                double insert = ((double)i / meanEnergy.length) * 289;
                sceneChangeTimesList.add(insert);
            }
        }

        int[] sceneChangeTimes = new int[sceneChangeTimesList.size()];
        String[] sceneChangeTimesString = new String[sceneChangeTimesList.size()];
        for (int i = 0; i < sceneChangeTimes.length; i++) {
            sceneChangeTimes[i] = sceneChangeTimesList.get(i).intValue();
            String ins = "";
            int min = (int)(sceneChangeTimesList.get(i) / 60);
            int remaining = (int)(sceneChangeTimesList.get(i) - ( 60 * min));
            ins += min + ":" + remaining;
            sceneChangeTimesString[i] = ins;
        }


        for (String time : sceneChangeTimesString) {
            System.out.println("Scene detected at " + time);
        }
        System.out.println(sceneChangeTimesString.length);


        return sceneChangeTimes;
    }
}