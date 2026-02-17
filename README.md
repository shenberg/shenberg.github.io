# shenberg.github.io

Source for a GitHub Pages blog powered by Jekyll.

## Local development

1. Install Ruby + Bundler.
2. Install dependencies:

   ```bash
   bundle install
   ```

3. Run locally:

   ```bash
   bundle exec jekyll serve
   ```

4. Open <http://127.0.0.1:4000>.

## Writing posts

- Add markdown files to `_posts/` with filename format `YYYY-MM-DD-title.md`.
- Use front matter with at least `title`, `date`, and `slug`.
- Post URLs are `/blog/:slug/`.

## Creative coding posts

- For small experiments, inline `<script>` directly in a post.
- If script content includes `{{ ... }}`, wrap that block in `{% raw %}` and `{% endraw %}`.
- For larger experiments, place JavaScript in `assets/js/...` and reference it from post front matter:

  ```yaml
  scripts:
    - /assets/js/my-sketch.js
  ```
