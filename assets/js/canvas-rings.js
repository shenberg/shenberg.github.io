(() => {
  const canvas = document.getElementById("rings-demo");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let frame = 0;
  const radius = Math.min(canvas.width, canvas.height) * 0.3;

  function draw() {
    frame += 1;
    ctx.fillStyle = "#0f1720";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < 6; i += 1) {
      const t = frame * 0.015 + i * 0.8;
      const x = canvas.width / 2 + Math.cos(t) * radius;
      const y = canvas.height / 2 + Math.sin(t * 1.3) * radius * 0.75;
      const size = 22 + Math.sin(t * 2) * 8;

      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.strokeStyle = `hsl(${(frame + i * 40) % 360}, 85%, 62%)`;
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    requestAnimationFrame(draw);
  }

  draw();
})();
