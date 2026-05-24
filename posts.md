---
layout: page
permalink: /
---

<div class="about-card">
  <div class="about-avatar">P</div>
  <div class="about-body">
    <p class="about-name">pluckh</p>
    <p class="about-bio">这是我的个人博客，记录技术、思考和生活。</p>
    <nav class="about-links">
      <a href="mailto:huangjiangbo.z@gmail.com">✉ Email</a>
      <a href="https://github.com/pluckhuang" target="_blank" rel="noopener">⚙ GitHub</a>
      <a href="https://douban.com/people/28360619" target="_blank" rel="noopener">◉ Douban</a>
    </nav>
  </div>
</div>

{% assign cat_list = "技术,云计算,思考,生活,文化" | split: "," %}

{% for cat in cat_list %}
  {% assign cat_posts = "" | split: "" %}
  {% for post in site.posts %}
    {% if post.tags contains cat %}
      {% assign cat_posts = cat_posts | push: post %}
    {% endif %}
  {% endfor %}
  {% if cat_posts.size > 0 %}
<section class="cat-section">
  <h2 class="cat-title">{{ cat }}<span class="cat-count">{{ cat_posts.size }}</span></h2>
  <div class="entries-list">
    {% for post in cat_posts %}
    <article class="entry h-entry">
      <header class="entry-header">
        <h3 class="entry-title p-name">
          <a href="{{ post.url | relative_url }}" rel="bookmark">{{ post.title }}</a>
        </h3>
      </header>
      <footer class="entry-meta">
        <time class="entry-date dt-published" datetime="{{ post.date | date_to_xmlschema }}">
          {{ post.date | date: "%Y-%m-%d" }}
        </time>
      </footer>
    </article>
    {% endfor %}
  </div>
</section>
  {% endif %}
{% endfor %}
