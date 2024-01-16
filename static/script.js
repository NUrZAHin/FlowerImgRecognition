document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("uploadForm");

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(form);

    axios.post("/", formData)
      .then(response => {
        updateResult(response.data);
      })
      .catch(error => {
        console.error("Error:", error.response.data.error);
      });
  });

  function updateResult(data) {
    const resultContainer = document.getElementById("resultContainer");
    const imageElement = document.getElementById("flowerImage");
    const resultText = document.getElementById("resultText");

    imageElement.src = data.image_path;

    resultText.textContent = `Predicted Flower Class: ${data.predicted_class}`;

    resultContainer.classList.remove("hidden");
  }
});