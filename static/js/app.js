/**
 * DSA Intelligence — Frontend Application Logic
 * Handles API calls, data rendering, animations, and UI interactions.
 */

// ── State ──
let currentData = null;
let currentRecommendations = null;
let allCompanies = [];

// ── Utility ──
function setUsername(name) {
    document.getElementById('username-input').value = name;
    document.getElementById('username-input').focus();
}

function showError(msg) {
    document.getElementById('error-message').textContent = msg;
    document.getElementById('error-container').style.display = 'block';
}

function hideError() {
    document.getElementById('error-container').style.display = 'none';
}

function setLoading(loading) {
    const btn = document.getElementById('analyze-btn');
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    btn.disabled = loading;
    text.style.display = loading ? 'none' : 'inline';
    loader.style.display = loading ? 'inline-flex' : 'none';
}

// Animated counter
function animateCounter(elementId, target, duration = 1200) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const start = 0;
    const startTime = performance.now();
    target = Math.round(target);

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ── Main Analyze Function ──
async function analyzeUser() {
    const username = document.getElementById('username-input').value.trim();
    if (!username) {
        showError('Please enter a LeetCode username');
        return;
    }

    hideError();
    setLoading(true);
    document.getElementById('results').style.display = 'none';
    document.getElementById('recommendations').style.display = 'none';

    try {
        // Fetch analysis
        const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });

        const data = await res.json();
        if (!res.ok) {
            showError(data.error || 'An error occurred');
            setLoading(false);
            return;
        }

        currentData = data;
        window.currentData = data; // Expose globally for chat.js

        // Fetch recommendations
        const recRes = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
        });

        const recData = await recRes.json();
        if (recRes.ok) {
            currentRecommendations = recData.recommendations;
        }

        // Render everything
        renderProfile(data);
        renderPredictions(data.prediction);
        renderTopicAnalysis(data.topic_analysis);
        renderStudyPlan(data.study_plan);
        renderPipelineDashboard(data.pipeline_info);
        renderRecommendations(currentRecommendations);

        document.getElementById('results').style.display = 'block';
        document.getElementById('recommendations').style.display = 'block';

        // Scroll to results
        setTimeout(() => {
            document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);

    } catch (err) {
        console.error(err);
        showError('Network error. Please check your connection and try again.');
    } finally {
        setLoading(false);
    }
}

// ── Render Profile ──
function renderProfile(data) {
    const user = data.user_data;
    const prediction = data.prediction;

    // Avatar
    const avatar = document.getElementById('profile-avatar');
    const avatarUrl = user.profile?.userAvatar;
    avatar.src = avatarUrl || `https://ui-avatars.com/api/?name=${user.username}&background=6366f1&color=fff&size=150`;

    // Username & ranking
    document.getElementById('profile-username').textContent = user.username;
    const ranking = user.ranking ? `Global Rank #${user.ranking.toLocaleString()}` : 'Unranked';
    document.getElementById('profile-ranking').textContent = ranking;

    // Skill badge
    const badge = document.getElementById('skill-badge');
    const level = prediction.skill_level.toLowerCase();
    badge.textContent = prediction.skill_level;
    badge.className = `profile-badge ${level}`;

    // Stats with animation
    animateCounter('stat-total', user.stats.total);
    animateCounter('stat-easy', user.stats.easy);
    animateCounter('stat-medium', user.stats.medium);
    animateCounter('stat-hard', user.stats.hard);
}

// ── Render Predictions ──
function renderPredictions(prediction) {
    // Skill gauge
    const skillIdx = prediction.skill_index; // 0-3
    const gaugePercent = ((skillIdx + 1) / 4) * 100;
    const gaugeFill = document.getElementById('gauge-fill');
    const arcLength = 251.2;
    const fillLength = (gaugePercent / 100) * arcLength;

    setTimeout(() => {
        gaugeFill.setAttribute('stroke-dasharray', `${fillLength} ${arcLength}`);
    }, 300);

    document.getElementById('gauge-label').textContent = prediction.skill_level;

    // Confidence bars
    const confContainer = document.getElementById('confidence-bars');
    confContainer.innerHTML = '';
    const levels = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];
    levels.forEach(level => {
        const val = prediction.confidence[level] || 0;
        const row = document.createElement('div');
        row.className = 'conf-bar-row';
        row.innerHTML = `
            <span class="conf-label">${level}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill ${level.toLowerCase()}" style="width: 0%"></div>
            </div>
            <span class="conf-value">${val.toFixed(1)}%</span>
        `;
        confContainer.appendChild(row);

        // Animate bar
        setTimeout(() => {
            row.querySelector('.conf-bar-fill').style.width = `${val}%`;
        }, 500);
    });

    // Readiness ring
    const readiness = prediction.placement_readiness;
    const circle = document.getElementById('readiness-circle');
    const circumference = 2 * Math.PI * 70; // 440

    setTimeout(() => {
        const offset = (readiness / 100) * circumference;
        circle.setAttribute('stroke-dasharray', `${offset} ${circumference}`);
    }, 400);

    // Animated readiness value
    const readinessEl = document.getElementById('readiness-value');
    const startTime = performance.now();
    function animateReadiness(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / 1500, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        readinessEl.textContent = Math.round(readiness * eased);
        if (progress < 1) requestAnimationFrame(animateReadiness);
    }
    requestAnimationFrame(animateReadiness);

    // Readiness badge
    const readinessBadge = document.getElementById('readiness-badge');
    let badgeColor, badgeText;
    if (readiness >= 80) { badgeColor = 'var(--accent-green)'; badgeText = '🚀 Highly Ready'; }
    else if (readiness >= 60) { badgeColor = 'var(--accent-cyan)'; badgeText = '📈 Moderately Ready'; }
    else if (readiness >= 40) { badgeColor = 'var(--accent-yellow)'; badgeText = '🔨 Developing'; }
    else if (readiness >= 20) { badgeColor = 'var(--accent-red)'; badgeText = '🌱 Early Stage'; }
    else { badgeColor = 'var(--text-muted)'; badgeText = '🏁 Just Starting'; }

    readinessBadge.style.background = `color-mix(in srgb, ${badgeColor} 15%, transparent)`;
    readinessBadge.style.color = badgeColor;
    readinessBadge.style.border = `1px solid color-mix(in srgb, ${badgeColor} 30%, transparent)`;
    readinessBadge.textContent = badgeText;

    // Contest stats
    renderContestStats();
}

function renderContestStats() {
    if (!currentData) return;
    const contest = currentData.user_data.contest;
    const container = document.getElementById('contest-stats');
    container.innerHTML = `
        <div class="contest-stat">
            <div class="contest-stat-value">${contest.rating || '—'}</div>
            <div class="contest-stat-label">Contest Rating</div>
        </div>
        <div class="contest-stat">
            <div class="contest-stat-value">${contest.attended || 0}</div>
            <div class="contest-stat-label">Contests</div>
        </div>
        <div class="contest-stat">
            <div class="contest-stat-value">${contest.top_percentage ? contest.top_percentage + '%' : '—'}</div>
            <div class="contest-stat-label">Top %</div>
        </div>
    `;
}

// ── Render Topic Analysis ──
function renderTopicAnalysis(analysis) {
    const strengthsList = document.getElementById('strengths-list');
    const weaknessesList = document.getElementById('weaknesses-list');

    strengthsList.innerHTML = '';
    weaknessesList.innerHTML = '';

    if (!analysis.strengths.length) {
        strengthsList.innerHTML = '<div class="topic-item"><span class="topic-name" style="color:var(--text-muted);">Solve more problems to see strengths</span></div>';
    }

    analysis.strengths.forEach((t, i) => {
        strengthsList.innerHTML += `
            <div class="topic-item" style="animation-delay:${i * 0.1}s">
                <div class="topic-rank strength">${i + 1}</div>
                <span class="topic-name">${t.name}</span>
                <span class="topic-level-badge ${t.level}">${t.level}</span>
                <span class="topic-count">${t.solved} solved</span>
            </div>
        `;
    });

    if (!analysis.weaknesses.length) {
        weaknessesList.innerHTML = '<div class="topic-item"><span class="topic-name" style="color:var(--text-muted);">Not enough data yet</span></div>';
    }

    analysis.weaknesses.forEach((t, i) => {
        weaknessesList.innerHTML += `
            <div class="topic-item" style="animation-delay:${i * 0.1}s">
                <div class="topic-rank weakness">${i + 1}</div>
                <span class="topic-name">${t.name}</span>
                <span class="topic-level-badge ${t.level}">${t.level}</span>
                <span class="topic-count">${t.solved} solved</span>
            </div>
        `;
    });

    // All topics chart
    renderTopicsChart(analysis.all_topics);
}

function renderTopicsChart(topics) {
    const container = document.getElementById('topics-chart');
    container.innerHTML = '';

    if (!topics || !topics.length) {
        container.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;">No topic data available</p>';
        return;
    }

    const maxSolved = Math.max(...topics.map(t => t.solved), 1);
    const colors = [
        '#6366f1', '#a855f7', '#ec4899', '#22d3ee',
        '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
        '#06b6d4', '#14b8a6', '#f97316', '#64748b'
    ];

    topics.forEach((t, i) => {
        const pct = (t.solved / maxSolved) * 100;
        const color = colors[i % colors.length];

        const item = document.createElement('div');
        item.className = 'topic-bar-item';
        item.innerHTML = `
            <span class="topic-bar-name" title="${t.name}">${t.name}</span>
            <div class="topic-bar-bg">
                <div class="topic-bar-fill" style="background:${color};width:0%"></div>
            </div>
            <span class="topic-bar-val">${t.solved}</span>
        `;
        container.appendChild(item);

        // Animate
        setTimeout(() => {
            item.querySelector('.topic-bar-fill').style.width = `${pct}%`;
        }, 300 + i * 50);
    });
}

// ── Render Study Plan ──
function renderStudyPlan(plan) {
    // Daily target
    animateCounter('daily-target', plan.daily_target, 800);

    // Focus areas
    const focusEl = document.getElementById('focus-areas');
    if (plan.focus_areas.length) {
        focusEl.innerHTML = `
            <div class="plan-focus-title">Focus Areas</div>
            <div class="focus-tags">
                ${plan.focus_areas.map(a => `<span class="focus-tag">${a}</span>`).join('')}
            </div>
        `;
    } else {
        focusEl.innerHTML = '<div class="plan-focus-title">Keep exploring different topics!</div>';
    }

    // Weekly plan
    const weeklyEl = document.getElementById('weekly-plan');
    weeklyEl.innerHTML = `
        <div class="schedule-title">📅 Weekly Schedule</div>
        <div class="schedule-items">
            ${plan.weekly_plan.map(item => `<div class="schedule-item">${item}</div>`).join('')}
        </div>
    `;

    // Milestones
    const milestonesEl = document.getElementById('milestones');
    milestonesEl.innerHTML = `
        <div class="milestones-title">🎯 Milestones</div>
        <div class="milestone-items">
            ${plan.milestones.map(m => `
                <div class="milestone-item">
                    <div class="milestone-icon">◎</div>
                    <span>${m}</span>
                </div>
            `).join('')}
        </div>
    `;
}

// ── Render Big Data Pipeline Dashboard ──
function renderPipelineDashboard(pipelineInfo) {
    if (!pipelineInfo) return;

    const { ml_metadata, db_stats, pipeline_summary } = pipelineInfo;

    // Engine
    document.getElementById('pipe-engine').textContent = 
        ml_metadata.engine.includes('Spark') ? 'Apache Spark' : 'scikit-learn';
    
    // Dataset size
    document.getElementById('pipe-records').textContent = 
        (ml_metadata.dataset_size || 0).toLocaleString();

    // Database
    document.getElementById('pipe-db').textContent = 
        db_stats.connected ? 'MongoDB' : 'In-Memory';

    // Processing Time
    const time = pipeline_summary.processing_time || ml_metadata.training_time_seconds;
    document.getElementById('pipe-time').textContent = 
        time ? `${time.toFixed(1)}s` : '—';

    // Accuracy
    document.getElementById('pipe-accuracy').textContent = 
        ml_metadata.skill_accuracy ? `${(ml_metadata.skill_accuracy * 100).toFixed(1)}%` : '—';

    // Analyses Count
    document.getElementById('pipe-analyses').textContent = 
        (db_stats.total_analyses || 1).toLocaleString();

    // Add highlight animation
    const dashboard = document.querySelector('.pipeline-dashboard');
    dashboard.style.boxShadow = '0 0 30px rgba(16, 185, 129, 0.4)';
    setTimeout(() => {
        dashboard.style.boxShadow = 'none';
    }, 1000);
}

// ── Render Recommendations ──
function renderRecommendations(recommendations) {
    if (!recommendations) return;

    allCompanies = Object.keys(recommendations);

    // Company filter buttons
    const btnContainer = document.getElementById('company-buttons');
    btnContainer.innerHTML = allCompanies.map(company =>
        `<button class="company-btn" data-company="${company}" onclick="filterCompany('${company}')">${company}</button>`
    ).join('');

    renderCompanyCards(recommendations);
}

function filterCompany(company) {
    // Update active button
    document.querySelectorAll('.company-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.company === company);
    });

    if (company === 'all') {
        renderCompanyCards(currentRecommendations);
    } else {
        const filtered = {};
        filtered[company] = currentRecommendations[company];
        renderCompanyCards(filtered);
    }
}

function renderCompanyCards(recommendations) {
    const grid = document.getElementById('recommendations-grid');
    grid.innerHTML = '';

    const companyEmojis = {
        'Google': '🔍', 'Amazon': '📦', 'Microsoft': '🪟', 'Meta': '👤',
        'Apple': '🍎', 'Goldman Sachs': '💰', 'Uber': '🚗', 'Adobe': '🎨'
    };

    for (const [company, questions] of Object.entries(recommendations)) {
        const card = document.createElement('div');
        card.className = 'company-card';

        const emoji = companyEmojis[company] || '🏢';
        const initial = company.charAt(0);

        card.innerHTML = `
            <div class="company-card-header">
                <div class="company-logo">${initial}</div>
                <div>
                    <div class="company-name">${emoji} ${company}</div>
                </div>
            </div>
            <div class="question-list">
                ${questions.map(q => `
                    <a href="${q.url}" target="_blank" rel="noopener" class="question-item">
                        <div class="question-diff ${q.difficulty.toLowerCase()}"></div>
                        <div class="question-info">
                            <div class="question-title">${q.title}</div>
                            <div class="question-topic">${q.topic} · ${q.difficulty}</div>
                        </div>
                        <svg class="question-link-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M7 17L17 7M17 7H7M17 7V17"/>
                        </svg>
                    </a>
                `).join('')}
            </div>
        `;

        grid.appendChild(card);
    }
}

// ── Enter key support ──
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('username-input');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') analyzeUser();
    });

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});
