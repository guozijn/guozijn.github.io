---
layout: page
title: Credentials
permalink: /credentials/
published: false
recognitions:
  - title: Calculus for Machine Learning and Data Science
    organization: DeepLearning.AI
    date: 2025
    image: https://images.zjguo.com/coursera-calculus.png
    link: https://www.coursera.org/account/accomplishments/verify/MEWNX7C6L28D
    summary: Gained practical knowledge of calculus concepts such as limits, derivatives, and integrals, and applied them to optimisation and machine learning problems.
  - title: Linear Algebra for Machine Learning and Data Science
    organization: DeepLearning.AI
    date: 2025
    image: https://images.zjguo.com/coursera-linear-algebra.png
    link: https://www.coursera.org/account/accomplishments/verify/2WFRYPUQ8YC7
    summary: Acquired a strong foundation in vectors, matrices, eigenvalues, and linear transformations, and learned how these tools support data science and machine learning algorithms.
  - title: AWS Certified Solutions Architect – Professional
    organization: Amazon Web Services
    date: 2018
    image: https://images.zjguo.com/aws-certified-solutions-architect-professional.png
    link: https://cp.certmetrics.com/amazon/en/public/verify/credential/6VZ1F5WCB1F4QJ5Q
    summary: Demonstrated expertise in architecting and deploying large-scale, distributed systems on AWS.
  - title: AWS Certified Solutions Architect – Associate
    organization: Amazon Web Services
    date: 2017
    image: https://images.zjguo.com/aws-certified-solutions-architect-associate.png
    link: https://cp.certmetrics.com/amazon/en/public/verify/credential/Y7QD5RTK1EFE1P9M
    summary: Validated the ability to design resilient, cost-optimized AWS workloads across core services.
---

{% assign items = page.recognitions %}
{% if items and items.size > 0 %}
<div class="cred-grid">
  {% for item in items %}
  <article class="cred-card">
    <a class="cred-card__thumb" href="{{ item.link }}" target="_blank" rel="noopener" aria-label="View {{ item.title }}">
      <img src="{{ item.image | relative_url }}" alt="{{ item.title }}" loading="lazy">
      <span class="cred-card__overlay" aria-hidden="true">Open ↗</span>
    </a>
    <div class="cred-card__body">
      <p class="cred-card__org">{{ item.organization }}{% if item.date %} · {{ item.date }}{% endif %}</p>
      <h3 class="cred-card__title">{{ item.title }}</h3>
      {% if item.summary %}
      <p class="cred-card__summary">{{ item.summary }}</p>
      {% endif %}
    </div>
  </article>
  {% endfor %}
</div>
{% endif %}
