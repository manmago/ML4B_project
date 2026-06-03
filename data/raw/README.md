# Raw data files

Die Sensordaten sind hier verfügbar:

https://drive.google.com/drive/folders/1aFBSIv5vDCq0KVVvf9WHo5APG2wJ_4VP?usp=sharing

data/ Ordner herunterladen, entpacken und in den ML Ordner legen.
Dann: python -m streamlit run app.py




Contains the exported folders from each recorded sleep session with SensorLogger. 
Each folder is one night, named in the format: "yyy-mm-dd_hh-mm-ss".
Inside are csv files:
- Accelorometer.csv
    - time,seconds_elapsed,z,y,x
- Annotation.csv
    - time,seconds_elapsed,text,millisecond_press_duration
- Gyroscope.csv
    - time,seconds_elapsed,z,y,x
- Metadata.csv
    - version,device name,recording epoch time,recording time,recording timezone,platform,appVersion,device id,sensors,sampleRateMs,standardisation,platform version
- TotalAcceleration.csv
    - time,seconds_elapsed,z,y,x

Text input from Annotation.csv is manually input into a new file, "manual-labels.csv", with the columns: "night_id,bed_time,wake_time"

Timestamps: Unix format (nanoseconds)
Timestamps manual-labels.csv: YYYY-MM-DD HH:MM:SS in GMT +2 (ISO 8601)  
Values: SI units (m/s² for acceleration, rad/s for angular velocity)

Content is ignored by git.
