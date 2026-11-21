const $ = s => document.querySelector(s);
let mode = "chat", history = [];

const labels = {
  chat: "AI Chat",
  image: "Image Studio",
  coding: "Coding Assistant",
  documents: "Documents"
};

function showView(name) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  const view = $("#" + name + "View");
  if (view) view.classList.add("active");

  document.querySelectorAll(".nav").forEach(b =>
    b.classList.toggle("active", b.dataset.view === name)
  );

  $("#pageName").textContent = labels[name] || "AI Chat";
  $("#sidebar").classList.remove("open");
}

document.querySelectorAll(".nav").forEach(b =>
  b.addEventListener("click", () => showView(b.dataset.view))
);

document.querySelectorAll("[data-goto]").forEach(b =>
  b.addEventListener("click", () => showView(b.dataset.goto))
);

$("#mobileMenu").addEventListener("click", () =>
  $("#sidebar").classList.toggle("open")
);

$("#newChat").addEventListener("click", () => {
  history = [];
  $("#chatLog").innerHTML = "";
  $("#message").value = "";
  showView("chat");
});

document.querySelectorAll(".mode").forEach(b =>
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    document.querySelectorAll(".mode").forEach(x =>
      x.classList.toggle("active", x === b)
    );
  })
);

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = "bubble " + role;
  el.textContent = text;
  $("#chatLog").appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return el;
}

// GROQ CHAT — بدون تغییر
$("#chatForm").addEventListener("submit", async e => {
  e.preventDefault();

  const message = $("#message").value.trim();
  if (!message) return;

  $("#message").value = "";
  bubble("user", message);

  const wait = bubble("assistant", "Thinking…");
  const send = $("#chatForm button");
  send.disabled = true;

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, mode, history })
    });

    const d = await r.json();
    if (!r.ok) throw Error(d.detail || "Request failed");

    wait.textContent = d.answer;

    history.push(
      { role: "user", text: message },
      { role: "assistant", text: d.answer }
    );

    history = history.slice(-20);

  } catch (err) {
    wait.textContent = "Error: " + err.message;
  } finally {
    send.disabled = false;
  }
});

// LOAD PUTER
let puterLoadPromise;

function loadPuter() {
  if (window.puter) return Promise.resolve(window.puter);
  if (puterLoadPromise) return puterLoadPromise;

  puterLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://js.puter.com/v2/";
    script.async = true;

    script.onload = () => {
      window.puter
        ? resolve(window.puter)
        : reject(new Error("Puter.js did not initialize"));
    };

    script.onerror = () =>
      reject(new Error("Could not load Puter.js. Check your internet connection."));

    document.head.appendChild(script);
  });

  return puterLoadPromise;
}

function fileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () =>
      reject(new Error("Could not read the uploaded image"));

    reader.readAsDataURL(file);
  });
}
// IMAGE UPLOAD
$("#imageFile").addEventListener("change", () => {
  $("#uploadTitle").textContent =
    $("#imageFile").files[0]?.name ||
    "Upload an image to edit (optional)";
});

// IMAGE STUDIO
$("#imageForm").addEventListener("submit", async e => {
  e.preventDefault();

  const prompt = $("#imagePrompt").value.trim();

  if (!prompt) {
    $("#imageResult").innerHTML =
      '<p class="error">Please describe the image or edit.</p>';
    return;
  }

  const btn = $("#imageSubmit");
  btn.disabled = true;
  btn.textContent = "Creating...";

  $("#imageResult").textContent =
    "Your image is being generated. Please wait...";

  const file = $("#imageFile").files[0];

  try {
    let resultBlob;

    try {
      // PUTER IMAGE GENERATION / EDITING
      const puter = await loadPuter();

      let result;

      if (file) {
        const imageData = await fileAsDataURL(file);

        result = await puter.ai.txt2img({
          prompt: prompt,
          input_image: imageData,
          model: "gpt-image-1"
        });
      } else {
        result = await puter.ai.txt2img(prompt, {
          model: "gpt-image-1"
        });
      }

      if (result instanceof Blob) {
        resultBlob = result;
      } else if (result instanceof HTMLImageElement) {
        const response = await fetch(result.src);
        resultBlob = await response.blob();
      } else {
        throw new Error("Puter returned an unsupported image format");
      }

    } catch (puterError) {
      console.warn("Puter failed; trying existing image API:", puterError);

      // FALLBACK: EXISTING POLLINATIONS / BACKEND
      const form = new FormData();
      form.append("prompt", prompt);

      if (file) {
        form.append("image_file", file);
      }

      const response = await fetch("/api/image", {
        method: "POST",
        body: form
      });

      if (!response.ok) {
        let data = {};
        try {
          data = await response.json();
        } catch {}

        throw new Error(
          data.detail ||
          "Puter and the existing image service both failed."
        );
      }

      resultBlob = await response.blob();
    }

    if (!resultBlob || !resultBlob.type.startsWith("image/")) {
      throw new Error("The service did not return a valid image.");
    }

    const url = URL.createObjectURL(resultBlob);

    $("#imageResult").innerHTML = "";

    const img = document.createElement("img");
    img.src = url;
    img.alt = "Generated image";
    img.style.maxWidth = "100%";
    img.style.height = "auto";

    $("#imageResult").appendChild(img);

    const a = document.createElement("a");
    a.href = url;
    a.download = "afg-ai-image.png";
    a.textContent = "Download image";
    a.className = "primary";

    $("#imageResult").appendChild(a);

  } catch (err) {
    $("#imageResult").innerHTML = "";

    const p = document.createElement("p");
    p.className = "error";
    p.textContent = err.message;

    $("#imageResult").appendChild(p);

  } finally {
    btn.disabled = false;
    btn.textContent = "✧ Generate image";
  }
});

// LANGUAGE
const lang = $("#language");

lang.addEventListener("change", () => {
  document.documentElement.lang = lang.value;

  document.documentElement.dir =
    lang.value === "fa" || lang.value === "ps"
      ? "rtl"
      : "ltr";

  document.body.classList.toggle(
    "rtl",
    document.documentElement.dir === "rtl"
  );
});
