package com.example.nostressmusic;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.media.MediaPlayer;
import android.os.Bundle;
import android.content.Intent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.io.IOException;


public class Screen2Activity extends AppCompatActivity {
    //    List<String> stressRelief = Arrays.asList("DNA", "MicDrop", "Fake Love","Memories", "Euphoria");
    List<String> stressRelief = Arrays.asList("Euphoria");
    List<String> normalSongs = Arrays.asList("Epiphany");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_screen2);
        TextView s;
        s = (TextView)findViewById(R.id.textView5);
        Button bb = (Button) findViewById(R.id.button2);
        System.out.println("TxtView" + s);
        Intent intent = getIntent();
        final String result = intent.getStringExtra("Prediction");
        int resultValue = Integer.parseInt(result);
        String str_to_set;
        final MediaPlayer mediaPlayer;

        if (resultValue == 1) {
            str_to_set = "You are Stressed! Listen to "+this.getRandomElement(stressRelief);
            s.setText(str_to_set);
            mediaPlayer = MediaPlayer.create(Screen2Activity.this, R.raw.euphoria);
            mediaPlayer.start();
        } else {
            str_to_set = "You are not stressed yet! Anyway listen to "+this.getRandomElement(normalSongs);
            s.setText(str_to_set);
            mediaPlayer = MediaPlayer.create(Screen2Activity.this, R.raw.epiphany);
            mediaPlayer.start();
        }
        bb.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v)
            {
                mediaPlayer.stop();
            }
        });
    }

    private String getRandomElement(List<String> list) {
        Random rand = new Random();
        return list.get(rand.nextInt(list.size()));
    }
}
