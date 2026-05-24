---
layout: page
permalink: /
---

<header class="site-hero">
  <div class="hero-name">pluckh</div>
  <p class="hero-bio">这是我的个人博客，记录技术、思考和生活。</p>
  <nav class="hero-links">
    <a href="mailto:huangjiangbo.z@gmail.com">Email</a>
    <a href="https://github.com/pluckhuang" target="_blank" rel="noopener">GitHub</a>
    <a href="https://douban.com/people/28360619" target="_blank" rel="noopener">Douban</a>
  </nav>
</header>

<section class="cat-section">
  <h2 class="cat-title">最近<span class="cat-count">{{ site.posts.size }}</span></h2>
  <div class="recent-list">
    {% for post in site.posts limit:6 %}
    <a href="{{ post.url | relative_url }}" class="recent-item">
      <span class="recent-title">{{ post.title }}</span>
      <span class="recent-meta">
        <span class="recent-tag">{{ post.tags | first }}</span>
        <time>{{ post.date | date: "%Y · %m" }}</time>
      </span>
    </a>
    {% endfor %}
  </div>
</section>

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
