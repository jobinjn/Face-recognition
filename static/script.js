const video = document.getElementById('video');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

function capture() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');

    fetch('/detect', {
        method: 'POST',
        body: new URLSearchParams({ image: dataURL }),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
    .then(res => res.json())
    .then(data => {
        console.log(data);
        loadAttendance();
    });
}

function loadAttendance() {
    fetch('/attendance')
        .then(res => res.json())
        .then(data => {
            const table = document.getElementById('attendanceTable');
            table.innerHTML = '<tr><th>Name</th><th>Time</th></tr>';
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td>${row.name}</td><td>${row.time}</td>`;
                table.appendChild(tr);
            });
        });
}

setInterval(loadAttendance, 5000);
