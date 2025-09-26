---
layout: page
title: Categories
permalink: /categories/
---


{% assign categories = site.categories | sort %}
<div class="tag-index">
  {% for category in categories %}
    <section id="{{ category[0] | slugify }}" class="tag-index__section">
      <h2 class="tag-index__title">{{ category[0] }}</h2>
      <ol class="tag-index__list">
        {% assign posts = category[1] | sort: 'date' | reverse %}
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
