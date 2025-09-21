---
layout: page
title: Tags
permalink: /tags/
---

{% assign tags = site.tags | sort %}
<div class="tag-index">
  {% for tag in tags %}
    <section id="{{ tag[0] | slugify }}" class="tag-index__section">
      <h2 class="tag-index__title">{{ tag[0] }}</h2>
      <ol class="tag-index__list">
        {% assign posts = tag[1] | sort: 'date' | reverse %}
        {% for post in posts %}
          <li>
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            <span class="tag-index__meta">{{ post.date | date: "%Y-%m-%d" }}</span>
          </li>
        {% endfor %}
      </ol>
    </section>
  {% endfor %}
</div>
