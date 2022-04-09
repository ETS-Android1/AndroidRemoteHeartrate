package com.example.mytest;

import android.util.Log;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.lang.Math;


public class WaveProcess {
    public static double[] WaveProcess(float[] input) {
        int low=0;
        int high=0;
//        //EMD
//        Emd.EmdData emd = Emd.decompose(input,2,5, 0);
//        float[] imfout = emd.imfs[1];
        //Envelop normalize
        float[] envline = getEnvelope(input);
        float[] normalized = floatarrayDivide(input,envline);

        //FFT
        Complex[] tmp = ArraytoComplex(normalized);
        Complex[] fftcompoutput = Fft.fft(tmp);
        double[] freq = fftshift(fftfreq(fftcompoutput.length, 1d/30d));
        double[] fftoutput = fftshift(complexAbs(fftcompoutput));

        //get peak
        //Log.i("In:", Integer.toString(freq.length)+" "+Integer.toString(fftoutput.length));
        for(int i=0; i<freq.length; i++){
            if(freq[i]>0.7&&low==0){
                low = i;
            }else if(freq[i] > 3.0){
                high = i - 1;
                break;
            }
        }
        double[] trimfreq=Arrays.copyOfRange(freq,low,high);
        double[] trimoutput=Arrays.copyOfRange(fftoutput,low,high);

        int[] peaks = findPeak(trimoutput, 2,15);
        double[] output = new double[peaks.length];
//        Log.i("Freq", "Freq List:");
        for(int i=0; i<peaks.length; i++){
            output[i] = trimfreq[peaks[i]];
//            Log.i("Freq", Double.toString(trimfreq[peaks[i]]));
        }
        if(peaks.length==0){
//            Log.i("Freq", "No peaks.");
        }

        return output;
    }

    //ref from: https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py
    public static Array2DRowRealMatrix whiteNoise(int dim, double dt){
        double[][] q;
        if(dim == 2){
            q = new double[][]{
                        {0.25d * Math.pow(dt, 4), 0.5d * Math.pow(dt, 3)}
                    ,   {0.5d * Math.pow(dt, 3) , Math.pow(dt, 2)}};
        }
        else if(dim == 3){
            q = new double[][]{
                        {0.25d * Math.pow(dt, 4),   0.5d * Math.pow(dt, 3), 0.5d * Math.pow(dt, 2)}
                    ,   {0.5d * Math.pow(dt, 3),    Math.pow(dt, 2),        dt}
                    ,   {0.5d * Math.pow(dt, 2),    dt,                     1}};
        }
        else {
            q = new double[][]{
                        {Math.pow(dt, 6) / 36,  Math.pow(dt, 5) / 12,   Math.pow(dt, 4) / 6,    Math.pow(dt, 3) / 6}
                    ,   {Math.pow(dt, 5) / 12,  Math.pow(dt, 4) / 4,    Math.pow(dt, 3) / 2,    Math.pow(dt, 2) / 2}
                    ,   {Math.pow(dt, 4) / 6,   Math.pow(dt, 3) / 2,    Math.pow(dt, 2),        dt}
                    ,   {Math.pow(dt, 3) / 6,   Math.pow(dt, 2) / 2,    dt,                     1}};
        }
        Array2DRowRealMatrix tmp = new Array2DRowRealMatrix(q);
        return tmp;
    }

    private static Complex[] ArraytoComplex(float[] input) {
        Complex[] tmp = new Complex[input.length];
        for (int i = 0; i < tmp.length; i++) {
            tmp[i] = new Complex(input[i], 0);
        }
        return tmp;
    }

    private static float[] getEnvelope(float[] input){
        float[] output = new float[input.length];
        ArrayList<Integer> u_x = new ArrayList<Integer>();
        ArrayList<Float> u_y = new ArrayList<Float>();
        ArrayList<Float> u_p = new ArrayList<Float>();
        u_x.add(0);
        u_y.add(input[0]);
        for(int i = 1; i < input.length - 1; i++){
            if(input[i] - input[i-1] > 0 && input[i] - input[i+1] > 0){
                u_x.add(i);
                u_y.add(input[i]);
            }
        }
        u_x.add(input.length-1);
        u_y.add(input[input.length-1]);
        //interpolate
        u_p.add(input[0]);
        for(int i = 1; i < u_x.size(); i++){
            float a = u_y.get(i - 1);
            float b = u_y.get(i);
            int end = u_x.get(i);
            for(int start = u_x.get(i - 1), j = u_x.get(i - 1) + 1; j < end; j++){
                float tmp = a + ((b - a) * (j - start) / (end - start));
                u_p.add(tmp);
            }
            u_p.add(b);
        }
        for(int i=0;i<input.length;i++){
            if(u_p.get(i)<input[i]){
                u_p.set(i,input[i]);
            }
        }

        return Floats.toArray(u_p);
    }

    private static float[] floatarrayDivide(float[] inu, float[] ind){
        float[] output = new float[inu.length];
        for(int i=0;i<inu.length;i++){
            output[i]=inu[i]/ind[i];
        }
        return output;
    }

    private static void printComplex(Complex[] input){
        for(int i = 0; i < input.length; i++){
            Log.i("Complex:",Double.toString(input[i].getReal())+"+"+Double.toString(input[i].getImaginary())+"i");
        }
    }
    private static void printfloat(float[] input){
        for(int i = 0; i < input.length; i++){
            Log.i("Float:",Float.toString(input[i]));
        }
    }
    private static void printdouble(double[] input){
        for(int i = 0; i < input.length; i++){
            Log.i("Double:",Double.toString(input[i]));
        }
    }

    private static double[] fftfreq(int n, double dt){
        ArrayList<Double> tmp = new ArrayList<Double>();
        if(n % 2 == 0){
            double dn = dt * n;
            for(int i = 0; i < n / 2; i++){
                tmp.add(i / dn);
            }
            for(int i = -1; i >= -n/2; i--) {
                tmp.add(i / dn);
            }
        }
        return Doubles.toArray(tmp);
    }

    private static double[] complexAbs(Complex[] input){
        ArrayList<Double> tmp = new ArrayList<Double>();
        for(int i=0; i < input.length; i++){
            tmp.add(Math.sqrt(Math.pow(input[i].getReal(),2)+Math.pow(input[i].getImaginary(),2)));
        }
        return Doubles.toArray(tmp);
    }

    private static double[] fftshift(double[] input){
        double[] tmp = new double[input.length];
        int shiftNum = input.length/2;
        for(int i=0; i<input.length; i++){
            tmp[(shiftNum + i)%input.length]=input[i];
        }
        return tmp;
    }

    private static int[] findPeak(double[] input, int peak_num, int iteration){
        ArrayList<Integer> tmp = new ArrayList<Integer>();
        double avg = 0;
        int iter = 0;
        for(int i = 1; i < input.length - 1; i++){
            avg+=input[i];
        }
        avg/=input.length;
        for(int i = 1; i < input.length - 1; i++){
            if(input[i] - input[i-1] > 0 && input[i] - input[i+1] > 0 && input[i] > avg){
                tmp.add(i);
            }
        }
        while (tmp.size()>peak_num && iter < iteration){
            avg *= 1.05;
            for(int i = tmp.size() - 1; i >= 0; i--){
                if(tmp.get(i) < avg){
                    tmp.remove(i);
                }
            }
        }
        return Ints.toArray(tmp);
    }

}
