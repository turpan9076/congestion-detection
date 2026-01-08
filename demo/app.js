let map;
let markers = [];
let times = [];
let cameras = [];
let currentIndex = 0;

fetch("demo_0101.json")
    .then(res => res.json())
    .then(data => {
        times = data.times;
        cameras = data.cameras;

        // スライダー設定
        const slider = document.getElementById("timeSlider");
        slider.max = times.length - 1;

        slider.addEventListener("input", () => {
        currentIndex = Number(slider.value);
        updateMap();
        });

        initMap();
        updateMap();
    });

window.addEventListener("resize", () => {
    if (map) {
        map.invalidateSize();
    }
});

function initMap() {
    map = L.map("map", {
        zoomControl: false,
        dragging: false,
        scrollWheelZoom: false,
        doubleClickZoom: false,
        boxZoom: false,
        keyboard: false,
        touchZoom: false
    }).setView([34.8, 134.7],12);

    L.tileLayer(
    "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
    {
        attribution: "地理院地図 © 国土地理院",
        maxZoom: 18
    }
    ).addTo(map);

    cameras.forEach(cam => {
        const marker = L.circleMarker([cam.lat, cam.lon], {
        radius: 8,
        color: "gray",
        fillOpacity: 0.8
        })
        .bindPopup(cam.name)
        .addTo(map);

        markers.push(marker);
    });
}

function updateMap() {
    // 時刻表示
    const label = document.getElementById("timeLabel");
    label.textContent = times[currentIndex].replace("T", " ");

    // 混雑率に応じて色変更
    cameras.forEach((cam, i) => {
        const value = cam.congestion[currentIndex];
        const marker = markers[i];

        if (value == null) {
        marker.setStyle({ color: "gray", fillColor: "gray" });
        } else if (value < 0.8) {
        marker.setStyle({ color: "green", fillColor: "green" });
        } else if (value < 1.0) {
        marker.setStyle({ color: "orange", fillColor: "orange" });
        } else {
        marker.setStyle({ color: "red", fillColor: "red" });
        }
    });
}
