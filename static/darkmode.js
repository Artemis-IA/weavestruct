// Log pour vérifier si le script est chargé
console.log("Dark mode script loaded");

// Fonction pour basculer entre le mode sombre et le mode clair
function toggleDarkMode() {
    const body = document.querySelector("body");
    const darkMode = body.classList.toggle("dark-mode");

    // Enregistrer la préférence dans le localStorage
    localStorage.setItem("swaggerDarkMode", darkMode);
}

// Fonction pour ajouter le toggle pour le mode sombre
function addDarkModeToggle() {
    const header = document.querySelector(".swagger-ui .topbar");
    if (!header) {
        console.warn("Swagger topbar not found, retrying...");
        return;
    }

    // Créer un conteneur pour le slider
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.alignItems = "center";
    container.style.gap = "10px";
    container.style.marginRight = "20px";
    container.style.padding = "10px";

    // Créer le slider toggle
    const sliderContainer = document.createElement("label");
    sliderContainer.style.cursor = "pointer";
    sliderContainer.style.display = "inline-flex";
    sliderContainer.style.alignItems = "center";
    sliderContainer.style.gap = "5px";

    const slider = document.createElement("input");
    slider.type = "checkbox";
    slider.style.display = "none";
    slider.onchange = toggleDarkMode;

    const sliderSpan = document.createElement("span");
    sliderSpan.style.width = "40px";
    sliderSpan.style.height = "20px";
    sliderSpan.style.background = "#ccc";
    sliderSpan.style.borderRadius = "20px";
    sliderSpan.style.position = "relative";
    sliderSpan.style.transition = "background 0.3s ease";

    const knob = document.createElement("span");
    knob.style.width = "18px";
    knob.style.height = "18px";
    knob.style.background = "#fff";
    knob.style.borderRadius = "50%";
    knob.style.position = "absolute";
    knob.style.top = "1px";
    knob.style.left = "1px";
    knob.style.transition = "all 0.3s ease";
    sliderSpan.appendChild(knob);

    slider.onchange = function () {
        toggleDarkMode();
        if (slider.checked) {
            sliderSpan.style.background = "#4caf50";
            knob.style.transform = "translateX(20px)";
        } else {
            sliderSpan.style.background = "#ccc";
            knob.style.transform = "translateX(0)";
        }
    };

    // Définir l'état initial à partir du localStorage
    if (localStorage.getItem("swaggerDarkMode") === "true") {
        document.querySelector("body").classList.add("dark-mode");
        slider.checked = true;
        sliderSpan.style.background = "#4caf50";
        knob.style.transform = "translateX(20px)";
    }

    // Ajouter les éléments du slider au conteneur
    sliderContainer.appendChild(slider);
    sliderContainer.appendChild(sliderSpan);

    const label = document.createElement("span");
    label.textContent = "Dark Mode";
    label.style.color = "#fff";

    container.appendChild(label);
    container.appendChild(sliderContainer);

    // Ajouter le slider à l'entête de Swagger
    header.appendChild(container);

    console.log("Dark mode toggle added successfully");
}

// Ajout du slider après le chargement complet du DOM
document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
        addDarkModeToggle();
    }, 1000); // Ajout d'un délai pour s'assurer que le DOM est complètement chargé
});
