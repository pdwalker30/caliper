const express = require("express");
const app = express();

app.get("/profile", (req, res) => {
  const name = req.query.name;
  const bio = req.query.bio;

  res.send(`
    <!DOCTYPE html>
    <html>
      <body>
        <h1>Welcome, ${name}!</h1>
        <div class="bio">${bio}</div>
      </body>
    </html>
  `);
});

app.listen(3000);
