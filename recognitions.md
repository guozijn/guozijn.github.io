---
layout: page
title: Recognitions
permalink: /recognitions/
description: My certificates, awards, and milestones.
recognitions:
  - title: Calculus for Machine Learning and Data Science
    organization: DeepLearning.AI
    date: 2025
    image: https://images.zijianguo.com/coursera-calculus.png
    link: https://www.coursera.org/account/accomplishments/verify/MEWNX7C6L28D
    summary: Gained practical knowledge of calculus concepts such as limits, derivatives, and integrals, and applied them to optimisation and machine learning problems.
  - title: Linear Algebra for Machine Learning and Data Science
    organization: DeepLearning.AI
    date: 2025
    image: https://images.zijianguo.com/coursera-linear-algebra.png
    link: https://www.coursera.org/account/accomplishments/verify/2WFRYPUQ8YC7
    summary: Acquired a strong foundation in vectors, matrices, eigenvalues, and linear transformations, and learned how these tools support data science and machine learning algorithms.
  - title: AWS Certified Solutions Architect – Professional
    organization: Amazon Web Services
    date: 2018
    image: https://images.zijianguo.com/aws-certified-solutions-architect-professional.png
    link: https://cp.certmetrics.com/amazon/en/public/verify/credential/6VZ1F5WCB1F4QJ5Q
    summary: Demonstrated expertise in architecting and deploying large-scale, distributed systems on AWS.
  - title: AWS Certified Solutions Architect – Associate
    organization: Amazon Web Services
    date: 2017
    image: https://images.zijianguo.com/aws-certified-solutions-architect-associate.png
    link: https://cp.certmetrics.com/amazon/en/public/verify/credential/Y7QD5RTK1EFE1P9M
    summary: Validated the ability to design resilient, cost-optimized AWS workloads across core services.
---

<div class="recognitions-intro">
  <p>Here is a snapshot of the credentials and milestones I am proud of. Each one reflects a phase of learning or collaboration that shaped my journey. Click any card to explore in more detail.</p>
</div>

{% assign items = page.recognitions %}
{% if items and items.size > 0 %}
<div class="recognitions-carousel" data-recognitions-carousel>
  <button class="recognitions-nav recognitions-nav--prev" type="button" aria-label="Previous">
    <span class="recognitions-nav__icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M15 5.5l-7 6.5 7 6.5" />
      </svg>
    </span>
  </button>
  <div class="recognitions-track" data-recognitions-track>
    {% for item in items %}
    <article class="recognition-card" data-recognition-card data-index="{{ forloop.index0 }}">
      <div class="recognition-thumb" data-image-wrapper>
        <img src="{{ item.image | relative_url }}" alt="{{ item.title }} certificate" loading="lazy" data-image-full="{{ item.image | relative_url }}">
      </div>
      <div class="recognition-body">
        <h3 class="recognition-title">{{ item.title }}</h3>
        <p class="recognition-meta">{{ item.organization }}{% if item.date %} · {{ item.date }}{% endif %}</p>
        {% if item.summary %}
        <p class="recognition-summary">{{ item.summary }}</p>
        {% endif %}
        {% if item.link %}
        <a class="recognition-link" href="{{ item.link }}" target="_blank" rel="noopener">View official link</a>
        {% endif %}
      </div>
    </article>
    {% endfor %}
  </div>
  <button class="recognitions-nav recognitions-nav--next" type="button" aria-label="Next">
    <span class="recognitions-nav__icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M9 5.5l7 6.5-7 6.5" />
      </svg>
    </span>
  </button>
</div>

<div class="recognition-modal" hidden data-recognitions-modal>
  <button class="recognition-modal__backdrop" type="button" data-close-modal aria-label="Close"></button>
  <div class="recognition-modal__dialog" role="dialog" aria-modal="true">
    <button class="recognition-modal__close" type="button" data-close-modal aria-label="Close">&times;</button>
    <img class="recognition-modal__image" data-modal-image src="" alt="">
    <p class="recognition-modal__caption" data-modal-caption></p>
  </div>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    const carousel = document.querySelector('[data-recognitions-carousel]');
    if (!carousel) return;

    const track = carousel.querySelector('[data-recognitions-track]');
    const cards = Array.from(track.querySelectorAll('[data-recognition-card]'));
    const prev = carousel.querySelector('.recognitions-nav--prev');
    const next = carousel.querySelector('.recognitions-nav--next');
    const modal = document.querySelector('[data-recognitions-modal]');
    const modalImage = modal ? modal.querySelector('[data-modal-image]') : null;
    const modalCaption = modal ? modal.querySelector('[data-modal-caption]') : null;
    const closeButtons = modal ? Array.from(modal.querySelectorAll('[data-close-modal]')) : [];

    const autoRotateDelay = 3000;
    const resumeDelay = 1000;

    if (cards.length === 0) return;

    let activeIndex = 0;
    let autoRotateTimer = null;
    let resumeTimer = null;

    function getGap() {
      const styles = window.getComputedStyle(track);
      return parseFloat(styles.columnGap || styles.gap || '0');
    }

    function getStep() {
      const first = cards[0];
      if (!first) return 0;
      return first.offsetWidth + getGap();
    }

    function clampIndex(index) {
      const maxIndex = Math.max(cards.length - 1, 0);
      if (index < 0) return 0;
      if (index > maxIndex) return maxIndex;
      return index;
    }

    function updateNav() {
      if (!prev || !next) return;
      const disable = cards.length <= 1;
      prev.disabled = disable;
      next.disabled = disable;
    }

    function scrollToIndex(index, smooth = true) {
      activeIndex = clampIndex(index);
      const offset = activeIndex * getStep();
      track.scrollTo({ left: offset, behavior: smooth ? 'smooth' : 'auto' });
      updateNav();
    }

    function move(delta, options = {}) {
      if (!cards.length) return;
      const loop = Boolean(options.loop);
      let target = activeIndex + delta;
      let smooth = options.smooth !== false;
      if (loop) {
        if (target < 0) {
          target = cards.length - 1;
          smooth = false;
        } else if (target >= cards.length) {
          target = 0;
          smooth = false;
        }
      }
      scrollToIndex(target, smooth);
    }

    function shouldPauseAuto() {
      if (cards.length <= 1) return true;
      if (modal && !modal.hasAttribute('hidden')) return true;
      const activeEl = document.activeElement;
      if (activeEl && carousel.contains(activeEl)) return true;
      if (carousel.matches(':hover')) return true;
      return false;
    }

    function stopAutoRotate() {
      if (autoRotateTimer !== null) {
        window.clearInterval(autoRotateTimer);
        autoRotateTimer = null;
      }
      if (resumeTimer !== null) {
        window.clearTimeout(resumeTimer);
        resumeTimer = null;
      }
    }

    function startAutoRotate(delay = autoRotateDelay) {
      stopAutoRotate();
      if (cards.length <= 1) return;
      if (shouldPauseAuto()) return;
      resumeTimer = window.setTimeout(function () {
        resumeTimer = null;
        if (shouldPauseAuto()) return;
        autoRotateTimer = window.setInterval(function () {
          move(1, { loop: true });
        }, autoRotateDelay);
      }, Math.max(delay, 0));
    }

    function openModal(imgSrc, caption) {
      if (!modal || !modalImage) return;
      stopAutoRotate();
      modalImage.src = imgSrc;
      modalImage.alt = caption || '';
      if (modalCaption) modalCaption.textContent = caption || '';
      modal.removeAttribute('hidden');
      document.body.classList.add('recognition-modal-open');
    }

    function closeModal() {
      if (!modal || !modalImage) return;
      modal.setAttribute('hidden', 'hidden');
      modalImage.src = '';
      document.body.classList.remove('recognition-modal-open');
      startAutoRotate(resumeDelay);
    }

    let resizeTimeout;
    window.addEventListener('resize', function () {
      window.clearTimeout(resizeTimeout);
      resizeTimeout = window.setTimeout(function () {
        scrollToIndex(activeIndex, false);
        startAutoRotate(resumeDelay);
      }, 150);
    });

    if (prev) {
      prev.addEventListener('click', function () {
        stopAutoRotate();
        move(-1, { loop: true });
        startAutoRotate(resumeDelay);
      });
    }

    if (next) {
      next.addEventListener('click', function () {
        stopAutoRotate();
        move(1, { loop: true });
        startAutoRotate(resumeDelay);
      });
    }

    cards.forEach(function (card, index) {
      card.addEventListener('mouseenter', function () {
        activeIndex = clampIndex(index);
        updateNav();
      });
      card.addEventListener('click', function (event) {
        if (event.target.closest('.recognition-link')) {
          return;
        }
        const img = card.querySelector('img[data-image-full]');
        if (!img) return;
        event.preventDefault();
        openModal(img.dataset.imageFull || img.src, img.alt);
      });
    });

    track.addEventListener('scroll', function () {
      const step = getStep();
      if (step <= 0) return;
      const newIndex = clampIndex(Math.round(track.scrollLeft / step));
      if (newIndex !== activeIndex) {
        activeIndex = newIndex;
        updateNav();
      }
    }, { passive: true });

    carousel.addEventListener('mouseenter', stopAutoRotate);
    carousel.addEventListener('mouseleave', function () { startAutoRotate(resumeDelay); });
    carousel.addEventListener('focusin', stopAutoRotate);
    carousel.addEventListener('focusout', function (event) {
      if (!carousel.contains(event.relatedTarget)) {
        startAutoRotate(resumeDelay);
      }
    });

    track.addEventListener('pointerdown', stopAutoRotate);
    track.addEventListener('pointerup', function () { startAutoRotate(resumeDelay); });
    track.addEventListener('wheel', function () {
      stopAutoRotate();
      startAutoRotate(resumeDelay);
    }, { passive: true });

    closeButtons.forEach(function (button) {
      button.addEventListener('click', closeModal);
    });

    if (modal) {
      modal.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') {
          closeModal();
        }
      });
    }

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') {
        closeModal();
      }
    });

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) {
        stopAutoRotate();
      } else {
        startAutoRotate(resumeDelay);
      }
    });

    updateNav();
    startAutoRotate(autoRotateDelay);
  });
</script>
{% else %}
<p>No recognitions are available yet. Add a <code>recognitions</code> list in the front matter to showcase your achievements.</p>
{% endif %}
