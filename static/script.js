// Main Application Script for AI Face Sentinel

let currentAttendanceData = [];
let regSource = 'file'; // 'file' or 'cam'
let webcamStream = null;
let snapshotDataUrl = null;

document.addEventListener("DOMContentLoaded", () => {
    // Initial data load
    loadStats();
    loadDates();
    loadFaceRegistry();
    loadActivityLogs();

    // Start background polling intervals
    setInterval(loadStats, 3000);
    setInterval(loadActivityLogs, 2500);
    setInterval(() => {
        const activeTab = document.querySelector(".nav-btn.active").getAttribute("onclick");
        if (activeTab && activeTab.includes("attendance")) {
            const dateSelect = document.getElementById("date-select");
            if (dateSelect && dateSelect.value) {
                loadAttendanceData(dateSelect.value, false);
            }
        }
    }, 4000);
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

    const selectedBtn = Array.from(document.querySelectorAll(".nav-btn")).find(btn => btn.getAttribute("onclick").includes(tabId));
    if (selectedBtn) selectedBtn.classList.add("active");

    const selectedTab = document.getElementById(`tab-${tabId}`);
    if (selectedTab) selectedTab.classList.add("active");

    if (tabId === 'attendance') {
        const dateSelect = document.getElementById("date-select");
        if (dateSelect.value) loadAttendanceData(dateSelect.value);
    } else if (tabId === 'registry') {
        loadFaceRegistry();
    }
}

// Fetch System Stats
async function loadStats() {
    try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("stat-registered").textContent = data.total_registered || 0;
        document.getElementById("stat-today").textContent = data.marked_today || 0;
        document.getElementById("stat-fake").textContent = data.fake_prevented || 0;
        
        const deviceEl = document.getElementById("stat-device");
        if (deviceEl) {
            deviceEl.textContent = data.device + (data.cuda_available ? " (GPU)" : "");
            deviceEl.style.color = data.cuda_available ? "var(--success)" : "var(--warning)";
        }
    } catch (err) {
        console.error("Error fetching stats:", err);
    }
}

// Fetch Live Logs
async function loadActivityLogs() {
    try {
        const res = await fetch("/api/logs");
        if (!res.ok) return;
        const logs = await res.json();

        const container = document.getElementById("activity-log-container");
        if (!container) return;

        if (logs.length === 0) {
            container.innerHTML = `<div class="log-item" style="color: var(--text-muted);">Listening for detection events...</div>`;
            return;
        }

        container.innerHTML = logs.map(log => {
            let badgeClass = "unknown";
            if (log.status === "REAL") badgeClass = "real";
            if (log.status === "FAKE") badgeClass = "fake";

            return `
                <div class="log-item">
                    <div>
                        <strong style="color: var(--text-primary);">${log.message}</strong>
                        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">${log.time}</div>
                    </div>
                    <span class="log-badge ${badgeClass}">${log.status}</span>
                </div>
            `;
        }).join("");
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

// Fetch Available Dates
async function loadDates() {
    try {
        const res = await fetch("/api/dates");
        const dates = await res.json();
        const select = document.getElementById("date-select");
        select.innerHTML = "";

        if (dates.length === 0) {
            const today = new Date().toISOString().split('T')[0];
            const opt = document.createElement("option");
            opt.value = today;
            opt.textContent = today + " (Today)";
            select.appendChild(opt);
            loadAttendanceData(today);
            return;
        }

        dates.forEach(date => {
            const opt = document.createElement("option");
            opt.value = date;
            opt.textContent = date;
            select.appendChild(opt);
        });

        select.value = dates[0];
        loadAttendanceData(dates[0]);
    } catch (err) {
        console.error("Error loading dates:", err);
    }
}

// Date Change Handler
function onDateChanged() {
    const date = document.getElementById("date-select").value;
    loadAttendanceData(date);
}

// Fetch Attendance Data for Date
async function loadAttendanceData(date, showLoading = true) {
    const tbody = document.getElementById("attendance-table-body");
    const downloadBtn = document.getElementById("download-btn");
    downloadBtn.href = `/download/${date}`;

    if (showLoading) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Loading logs for ${date}...</td></tr>`;
    }

    try {
        const res = await fetch(`/api/attendance/${date}`);
        currentAttendanceData = await res.json();
        renderAttendanceTable(currentAttendanceData);
    } catch (err) {
        console.error("Error loading attendance:", err);
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--danger);">Failed to load records.</td></tr>`;
    }
}

// Render Attendance Table
function renderAttendanceTable(records) {
    const tbody = document.getElementById("attendance-table-body");
    if (!records || records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No attendance marked for this date.</td></tr>`;
        return;
    }

    tbody.innerHTML = records.map((rec, index) => `
        <tr>
            <td style="color: var(--text-muted);">${index + 1}</td>
            <td style="font-weight: 600; color: var(--text-primary);">${rec.name}</td>
            <td><i class="fa-regular fa-clock" style="color: var(--accent-primary); margin-right: 6px;"></i>${rec.time}</td>
            <td><span class="log-badge real">VERIFIED</span></td>
        </tr>
    `).join("");
}

// Filter Attendance Table by Search Input
function filterAttendanceTable() {
    const query = document.getElementById("search-input").value.toLowerCase();
    const filtered = currentAttendanceData.filter(r => r.name.toLowerCase().includes(query));
    renderAttendanceTable(filtered);
}

// Fetch Registered Face Registry Grid
async function loadFaceRegistry() {
    const grid = document.getElementById("face-registry-grid");
    try {
        const res = await fetch("/api/faces");
        const faces = await res.json();

        if (faces.length === 0) {
            grid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 40px 0;">
                    <i class="fa-solid fa-folder-open" style="font-size: 2.5rem; margin-bottom: 12px; color: var(--text-muted);"></i>
                    <div>No registered faces found. Click 'Register New Face' to add users.</div>
                </div>
            `;
            return;
        }

        grid.innerHTML = faces.map(face => `
            <div class="face-card">
                <img src="${face.thumbnail_url}" class="face-thumb" alt="${face.name}" onerror="this.src='https://via.placeholder.com/100?text=Face'">
                <div class="face-name">${face.name}</div>
                <button class="btn btn-danger" style="padding: 6px 12px; font-size: 0.78rem; width: 100%; justify-content: center;" onclick="deleteFace('${face.name}')">
                    <i class="fa-solid fa-trash"></i> Delete
                </button>
            </div>
        `).join("");
    } catch (err) {
        console.error("Error loading registry:", err);
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--danger);">Failed to load registry.</div>`;
    }
}

// Delete Face
async function deleteFace(name) {
    if (!confirm(`Are you sure you want to delete '${name}' from face registry?`)) return;

    try {
        const res = await fetch(`/api/faces/${name}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            loadFaceRegistry();
            loadStats();
        } else {
            alert(data.message || "Failed to delete face.");
        }
    } catch (err) {
        console.error("Error deleting face:", err);
    }
}

// Modal Controls
function openRegisterModal() {
    document.getElementById("register-modal").classList.add("active");
    document.getElementById("reg-name").value = "";
    document.getElementById("reg-file").value = "";
    document.getElementById("reg-message").style.display = "none";
    setRegSource('file');
}

function closeRegisterModal() {
    document.getElementById("register-modal").classList.remove("active");
    stopWebcamStream();
}

function setRegSource(source) {
    regSource = source;
    const btnFile = document.getElementById("src-btn-file");
    const btnCam = document.getElementById("src-btn-cam");
    const containerFile = document.getElementById("reg-file-container");
    const containerCam = document.getElementById("reg-cam-container");

    if (source === 'file') {
        btnFile.classList.add("btn-primary");
        btnFile.style.background = "var(--accent-primary)";
        btnFile.style.color = "#FFF";
        btnCam.classList.remove("btn-primary");
        btnCam.style.background = "rgba(255,255,255,0.05)";
        btnCam.style.color = "var(--text-secondary)";

        containerFile.style.display = "block";
        containerCam.style.display = "none";
        stopWebcamStream();
    } else {
        btnCam.classList.add("btn-primary");
        btnCam.style.background = "var(--accent-primary)";
        btnCam.style.color = "#FFF";
        btnFile.classList.remove("btn-primary");
        btnFile.style.background = "rgba(255,255,255,0.05)";
        btnFile.style.color = "var(--text-secondary)";

        containerFile.style.display = "none";
        containerCam.style.display = "block";
        startWebcamStream();
    }
}

// Webcam Snapshot Logic
async function startWebcamStream() {
    const video = document.getElementById("reg-webcam-preview");
    const snapshotImg = document.getElementById("reg-snapshot-preview");
    snapshotImg.style.display = "none";
    video.style.display = "block";
    snapshotDataUrl = null;

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = webcamStream;
    } catch (err) {
        console.error("Camera access error:", err);
        const msg = document.getElementById("reg-message");
        msg.style.display = "block";
        msg.style.color = "var(--danger)";
        msg.textContent = "Could not access webcam for registration snapshot.";
    }
}

function stopWebcamStream() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

function takeSnapshot() {
    const video = document.getElementById("reg-webcam-preview");
    const canvas = document.getElementById("reg-canvas");
    const snapshotImg = document.getElementById("reg-snapshot-preview");

    if (!video.srcObject) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    snapshotDataUrl = canvas.toDataURL("image/jpeg");
    snapshotImg.src = snapshotDataUrl;
    snapshotImg.style.display = "block";
    video.style.display = "none";

    stopWebcamStream();
}

// Submit Face Registration
async function submitFaceRegistration() {
    const name = document.getElementById("reg-name").value.trim();
    const msg = document.getElementById("reg-message");
    msg.style.display = "block";

    if (!name) {
        msg.style.color = "var(--danger)";
        msg.textContent = "Please enter a person name.";
        return;
    }

    const formData = new FormData();
    formData.append("name", name);

    if (regSource === 'file') {
        const fileInput = document.getElementById("reg-file");
        if (!fileInput.files || fileInput.files.length === 0) {
            msg.style.color = "var(--danger)";
            msg.textContent = "Please select an image file.";
            return;
        }
        formData.append("image", fileInput.files[0]);
    } else {
        if (!snapshotDataUrl) {
            msg.style.color = "var(--danger)";
            msg.textContent = "Please click 'Capture Snapshot' first.";
            return;
        }
        formData.append("image_data", snapshotDataUrl);
    }

    msg.style.color = "var(--warning)";
    msg.textContent = "Processing face encoding... Please wait.";

    try {
        const res = await fetch("/api/register_face", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            msg.style.color = "var(--success)";
            msg.textContent = data.message;
            setTimeout(() => {
                closeRegisterModal();
                loadFaceRegistry();
                loadStats();
            }, 1000);
        } else {
            msg.style.color = "var(--danger)";
            msg.textContent = data.message || "Failed to register face.";
        }
    } catch (err) {
        console.error("Registration error:", err);
        msg.style.color = "var(--danger)";
        msg.textContent = "Server error during registration.";
    }
}
