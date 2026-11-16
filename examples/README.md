Verification clips recorded by the team for this project (shareable).

event_lamp_fire.mp4         oil lamp lit at about 5 s; expected: one fire alarm at about 6.6 s
normal_mountain_clouds.mp4  hills with low cloud; expected: no alarm

Both were cut from the team's phone recordings (clips/own/) and re-encoded to 1280x720, 30 fps,
H.264 with: ffmpeg -i <original> [-t 16] -vf "scale=1280:-2,fps=30" -c:v libx264 -preset veryfast -crf 26 -an
The outputs produced by infer.py are in ../expected_outputs/.
