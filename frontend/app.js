// ================================================================
// ADSA v2 — Frontend Application Logic
// SSE-based pipeline streaming, chat, chart gallery
// Navigation, scroll animations, and interactive UI
// ================================================================

// --- DOM References ---
const uploadForm     = document.getElementById('uploadForm');
const fileInput      = document.getElementById('fileInput');
const dropText       = document.getElementById('dropText');
const fileDropZone   = document.getElementById('fileDropZone');
const goalInput      = document.getElementById('goalInput');
const startBtn       = document.getElementById('startBtn');
const btnLabel       = document.getElementById('btnLabel');
const btnSpinner     = document.getElementById('btnSpinner');
const errorToast     = document.getElementById('errorToast');

const uploadSection   = document.getElementById('uploadSection');
const pipelineSection = document.getElementById('pipelineSection');
const resultsSection  = document.getElementById('resultsSection');

const pipelineTrack    = document.getElementById('pipelineTrack');
const liveBadge        = document.getElementById('liveBadge');
const pipelineSubtitle = document.getElementById('pipelineSubtitle');
const stepDetailCard   = document.getElementById('stepDetailCard');
const stepDetailTitle  = document.getElementById('stepDetailTitle');
const stepDetailBadge  = document.getElementById('stepDetailBadge');
const stepDetailBody   = document.getElementById('stepDetailBody');

const metricsGrid      = document.getElementById('metricsGrid');
const insightList      = document.getElementById('insightList');
const chartGallery     = document.getElementById('chartGallery');
const resultsSummary   = document.getElementById('resultsSummary');
const downloadReportBtn = document.getElementById('downloadReportBtn');
const newAnalysisBtn   = document.getElementById('newAnalysisBtn');

const chatMessages = document.getElementById('chatMessages');
const chatForm     = document.getElementById('chatForm');
const chatInput    = document.getElementById('chatInput');
const chatSendBtn  = document.getElementById('chatSendBtn');

// Navigation
const navbar     = document.getElementById('navbar');
const navToggle  = document.getElementById('navToggle');
const navLinks   = document.getElementById('navLinks');

// --- State ---
let sessionId = null;

const STEPS = [
    { id: 'planner',        label: 'Planner' },
    { id: 'profiler',       label: 'Profiler' },
    { id: 'cleaner',        label: 'Cleaner' },
    { id: 'eda',            label: 'EDA' },
    { id: 'visualizer',     label: 'Visualizer' },
    { id: 'feature_eng',    label: 'Features' },
    { id: 'modeling',       label: 'Modeling' },
    { id: 'hpo',            label: 'HPO' },
    { id: 'evaluation',     label: 'Evaluator' },
    { id: 'explainability', label: 'Explainer' },
    { id: 'insights',       label: 'Insights' },
    { id: 'report',         label: 'Reporter' },
];

const STEP_DESCRIPTIONS = {
    planner:        'Analyzing your goal and detecting task type...',
    profiler:       'Profiling dataset: columns, types, missing values...',
    cleaner:        'Cleaning data: imputing missing values, capping outliers...',
    eda:            'Running exploratory data analysis...',
    visualizer:     'Generating charts and visualizations...',
    feature_eng:    'Engineering features: encoding, scaling, selection...',
    modeling:       'Training multiple ML models with cross-validation...',
    hpo:            'Optimizing hyperparameters with Bayesian search...',
    evaluation:     'Evaluating model on holdout set...',
    explainability: 'Computing SHAP feature importance...',
    insights:       'Generating business insights...',
    report:         'Building PDF report...',
};

// ================================================================
// NAVIGATION
// ================================================================

// Scroll effect for navbar
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;

    // Add/remove scrolled class
    if (scrollY > 30) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }

    lastScroll = scrollY;

    // Update active nav link based on scroll position
    updateActiveNavLink();
}, { passive: true });

// Active nav link tracking
function updateActiveNavLink() {
    const sections = document.querySelectorAll('.hero-section, .section');
    const navLinksAll = document.querySelectorAll('.nav-link:not(.nav-link-cta)');
    let current = '';

    sections.forEach(section => {
        const top = section.offsetTop - 120;
        if (window.scrollY >= top) {
            current = section.getAttribute('id');
        }
    });

    navLinksAll.forEach(link => {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        if (href === `#${current}`) {
            link.classList.add('active');
        }
    });
}

// Mobile menu toggle
navToggle.addEventListener('click', () => {
    navToggle.classList.toggle('open');
    navLinks.classList.toggle('open');
});

// Close mobile menu on link click
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        navToggle.classList.remove('open');
        navLinks.classList.remove('open');
    });
});

// ================================================================
// SCROLL ANIMATIONS (Intersection Observer)
// ================================================================

function initScrollAnimations() {
    // Add fade-up class to animatable elements
    const animateTargets = document.querySelectorAll(
        '.feature-card, .how-step, .tech-item, .section-header'
    );

    animateTargets.forEach(el => {
        el.classList.add('fade-up');
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.dataset.delay || 0;
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, delay);
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    animateTargets.forEach(el => observer.observe(el));
}

// ================================================================
// HERO PARTICLES
// ================================================================

function initHeroParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;

    const particleCount = 30;
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 3 + 1}px;
            height: ${Math.random() * 3 + 1}px;
            border-radius: 50%;
            background: rgba(88, 166, 255, ${Math.random() * 0.3 + 0.05});
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: floatParticle ${Math.random() * 15 + 10}s ease-in-out infinite;
            animation-delay: ${Math.random() * 5}s;
        `;
        container.appendChild(particle);
    }

    // Add the float animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatParticle {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
            25% { transform: translate(${Math.random() * 60 - 30}px, ${Math.random() * 60 - 30}px) scale(1.2); opacity: 0.6; }
            50% { transform: translate(${Math.random() * 80 - 40}px, ${Math.random() * 80 - 40}px) scale(0.8); opacity: 0.4; }
            75% { transform: translate(${Math.random() * 60 - 30}px, ${Math.random() * 60 - 30}px) scale(1.1); opacity: 0.5; }
        }
    `;
    document.head.appendChild(style);
}

// ================================================================
// SMOOTH SCROLL FOR CTA BUTTONS
// ================================================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ================================================================
// FILE DROP ZONE
// ================================================================

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        dropText.textContent = fileInput.files[0].name;
        fileDropZone.classList.add('has-file');
    }
});

fileDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileDropZone.classList.add('dragover');
});

fileDropZone.addEventListener('dragleave', () => {
    fileDropZone.classList.remove('dragover');
});

fileDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    fileDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        dropText.textContent = e.dataTransfer.files[0].name;
        fileDropZone.classList.add('has-file');
    }
});

// ================================================================
// UPLOAD & START
// ================================================================

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorToast.classList.add('hidden');

    if (!fileInput.files.length) {
        showError('Please select a dataset file.');
        return;
    }

    // Loading state
    btnLabel.textContent = 'Uploading...';
    btnSpinner.classList.remove('hidden');
    startBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('user_goal', goalInput.value);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        sessionId = data.session_id;

        // Transition to pipeline view
        showPhase('pipeline');
        initPipelineTrack();
        startPipelineSSE();

    } catch (err) {
        showError(err.message);
    } finally {
        btnLabel.textContent = 'Launch Analysis';
        btnSpinner.classList.add('hidden');
        startBtn.disabled = false;
    }
});

// ================================================================
// PIPELINE SSE
// ================================================================

function startPipelineSSE() {
    const evtSource = new EventSource(`/api/run/${sessionId}`);

    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.status === 'running') {
            updateStepChip(data.step, 'running');
            updateStepDetail(data.step, STEP_DESCRIPTIONS[data.step] || 'Processing...');
        } else if (data.status === 'done') {
            updateStepChip(data.step, 'done');
            updateStepDetail(data.step, formatStepResult(data));
        } else if (data.status === 'error') {
            updateStepChip(data.step, 'error');
            updateStepDetail(data.step, `Error: ${data.error}`);
        } else if (data.status === 'completed') {
            evtSource.close();
            pipelineSubtitle.textContent = 'All agents completed successfully!';
            liveBadge.innerHTML = '<span>DONE</span>';
            liveBadge.classList.add('completed');

            // Show results after brief delay
            setTimeout(() => showResults(data), 800);
        }
    };

    evtSource.onerror = () => {
        evtSource.close();
        pipelineSubtitle.textContent = 'Connection lost. Check the server.';
    };
}

// ================================================================
// PIPELINE TRACK UI
// ================================================================

function initPipelineTrack() {
    pipelineTrack.innerHTML = '';
    STEPS.forEach((step) => {
        const chip = document.createElement('div');
        chip.className = 'step-chip';
        chip.id = `chip-${step.id}`;
        chip.innerHTML = `
            <span class="chip-icon"></span>
            <span>${step.label}</span>
        `;
        pipelineTrack.appendChild(chip);
    });
}

function updateStepChip(stepId, status) {
    const chip = document.getElementById(`chip-${stepId}`);
    if (!chip) return;

    chip.className = `step-chip ${status}`;
    const icon = chip.querySelector('.chip-icon');

    if (status === 'done') {
        icon.innerHTML = '✓';
    } else if (status === 'error') {
        icon.innerHTML = '✕';
    } else {
        icon.innerHTML = '';
    }
}

function updateStepDetail(stepId, content) {
    const step = STEPS.find(s => s.id === stepId);
    stepDetailTitle.textContent = step ? step.label : stepId;

    if (typeof content === 'string') {
        stepDetailBody.innerHTML = `<p>${content}</p>`;
        stepDetailBadge.textContent = 'In Progress';
        stepDetailBadge.className = 'detail-badge';
    } else {
        stepDetailBadge.textContent = 'Done';
        stepDetailBadge.className = 'detail-badge done';
    }
}

function formatStepResult(data) {
    const parts = [];
    if (data.task_type)        parts.push(`Task: <code>${data.task_type}</code>`);
    if (data.task_confidence)  parts.push(`Confidence: <code>${(data.task_confidence * 100).toFixed(0)}%</code>`);
    if (data.target_column)    parts.push(`Target: <code>${data.target_column}</code>`);
    if (data.rows !== undefined) parts.push(`${data.rows} rows × ${data.columns} columns`);
    if (data.duplicates)       parts.push(`${data.duplicates} duplicates found`);
    if (data.actions)          parts.push(`${data.actions} cleaning actions performed`);
    if (data.num_numeric !== undefined)  parts.push(`${data.num_numeric} numeric, ${data.num_categorical} categorical features`);
    if (data.charts)           parts.push(`${data.charts} charts generated`);
    if (data.features)         parts.push(`${data.features} features selected`);
    if (data.best_model)       parts.push(`Best model: <code>${data.best_model}</code>`);
    if (data.cv_results) {
        const cvText = Object.entries(data.cv_results)
            .map(([k, v]) => `${k}: ${v.toFixed(4)}`).join(', ');
        parts.push(`CV: ${cvText}`);
    }
    if (data.metrics) {
        const mText = Object.entries(data.metrics)
            .map(([k, v]) => `${k}: ${v}`).join(', ');
        parts.push(mText);
    }
    if (data.top_features) parts.push(`Top: ${data.top_features.join(', ')}`);
    if (data.insights)     parts.push(`${data.insights.length} insights generated`);
    if (data.report_ready) parts.push('PDF report ready for download');

    return parts.join('<br>') || 'Completed';
}

// ================================================================
// RESULTS
// ================================================================

function showResults(data) {
    showPhase('results');

    // Summary
    resultsSummary.textContent = `Best model: ${data.best_model || 'N/A'}`;

    // Metrics cards
    metricsGrid.innerHTML = '';
    const metrics = data.metrics || {};
    Object.entries(metrics).forEach(([key, value]) => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        const displayVal = typeof value === 'number'
            ? (value < 1 ? (value * 100).toFixed(1) + '%' : value.toFixed(4))
            : value;
        card.innerHTML = `
            <div class="metric-label">${key.replace(/_/g, ' ')}</div>
            <div class="metric-value">${displayVal}</div>
        `;
        metricsGrid.appendChild(card);
    });

    // Add best model card
    if (data.best_model) {
        const modelCard = document.createElement('div');
        modelCard.className = 'metric-card';
        modelCard.innerHTML = `
            <div class="metric-label">Best Model</div>
            <div class="metric-value" style="font-size:1.2rem">${data.best_model}</div>
        `;
        metricsGrid.prepend(modelCard);
    }

    // Insights
    insightList.innerHTML = '';
    (data.insights || []).forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        insightList.appendChild(li);
    });

    // Charts gallery
    chartGallery.innerHTML = '';
    (data.charts || []).forEach(filename => {
        const thumb = document.createElement('div');
        thumb.className = 'chart-thumb';
        const label = filename.replace('.png', '').replace(/_/g, ' ');
        thumb.innerHTML = `
            <img src="/api/charts/${filename}" alt="${label}" loading="lazy">
            <div class="chart-name">${label}</div>
        `;
        thumb.addEventListener('click', () => openLightbox(`/api/charts/${filename}`));
        chartGallery.appendChild(thumb);
    });
}

// ================================================================
// CHART LIGHTBOX
// ================================================================

function openLightbox(src) {
    const overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';
    overlay.innerHTML = `<img src="${src}" alt="Chart">`;
    overlay.addEventListener('click', () => overlay.remove());
    document.body.appendChild(overlay);
}

// ================================================================
// CHAT
// ================================================================

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question || !sessionId) return;

    // Add user message
    addChatMessage(question, 'user');
    chatInput.value = '';
    chatSendBtn.disabled = true;

    try {
        const res = await fetch(`/api/chat/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const data = await res.json();
        addChatMessage(data.answer || 'No response.', 'bot');
    } catch (err) {
        addChatMessage('Failed to get response. Please try again.', 'bot');
    } finally {
        chatSendBtn.disabled = false;
    }
});

function addChatMessage(text, sender) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg ${sender}-msg`;
    msgDiv.innerHTML = `
        <div class="msg-avatar">${sender === 'bot' ? 'AI' : 'You'}</div>
        <div class="msg-content"><p>${text}</p></div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ================================================================
// REPORT DOWNLOAD
// ================================================================

downloadReportBtn.addEventListener('click', () => {
    if (!sessionId) return;
    window.open(`/api/report/${sessionId}`, '_blank');
});

// ================================================================
// NEW ANALYSIS
// ================================================================

newAnalysisBtn.addEventListener('click', () => {
    sessionId = null;
    fileInput.value = '';
    goalInput.value = '';
    dropText.textContent = 'Drag & drop CSV or Excel file';
    fileDropZone.classList.remove('has-file');
    showPhase('upload');
});

// ================================================================
// PHASE NAVIGATION
// ================================================================

function showPhase(phase) {
    uploadSection.classList.add('hidden');
    pipelineSection.classList.add('hidden');
    resultsSection.classList.add('hidden');

    if (phase === 'upload')   uploadSection.classList.remove('hidden');
    if (phase === 'pipeline') pipelineSection.classList.remove('hidden');
    if (phase === 'results')  resultsSection.classList.remove('hidden');

    // Scroll to app section
    const appSection = document.getElementById('app');
    if (appSection) {
        appSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ================================================================
// ERROR DISPLAY
// ================================================================

function showError(msg) {
    errorToast.textContent = msg;
    errorToast.classList.remove('hidden');
    setTimeout(() => errorToast.classList.add('hidden'), 5000);
}

// ================================================================
// INITIALIZATION
// ================================================================

document.addEventListener('DOMContentLoaded', () => {
    initScrollAnimations();
    initHeroParticles();
    updateActiveNavLink();
});
