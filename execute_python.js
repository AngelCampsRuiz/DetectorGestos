const { exec } = require('child_process');

const videoUrl = 'https://www.youtube.com/watch?v=ocJTV1M6-9o&ab_channel=AlejaviRivera';
const extractTime = '01:30:500';

exec(`python3 analizador.py "${videoUrl}" --extract ${extractTime}`, (error, stdout, stderr) => {
    if (error) {
        console.error(`Error: ${error.message}`);
        return;
    }
    if (stderr) {
        console.error(`stderr: ${stderr}`);
        return;
    }
    console.log(`stdout: ${stdout}`);
});
