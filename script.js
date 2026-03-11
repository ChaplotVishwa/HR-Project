document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            tabContents.forEach(tab => {
                tab.classList.remove('active');
                if (tab.id === tabId) tab.classList.add('active');
            });
        });
    });

    // Helper functions
    const loader = document.getElementById('loader');
    const showLoader = () => loader.classList.remove('hidden');
    const hideLoader = () => loader.classList.add('hidden');

    const API_BASE = window.location.origin;

    // Show a non-blocking error message instead of alert
    function showError(message) {
        hideLoader();
        const errDiv = document.createElement('div');
        errDiv.style.cssText = 'position:fixed;top:20px;right:20px;background:#ef4444;color:white;padding:1rem 1.5rem;border-radius:10px;z-index:9999;max-width:400px;word-break:break-word;box-shadow:0 4px 12px rgba(0,0,0,0.3);';
        errDiv.textContent = message;
        document.body.appendChild(errDiv);
        setTimeout(() => errDiv.remove(), 6000);
    }

    // --- Unified Pipeline Form ---
    const unifiedForm = document.getElementById('unified-form');
    if (unifiedForm) {
        unifiedForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoader();

            const jd = document.getElementById('unified-jd').value;
            const fileInput = document.getElementById('unified-files');

            if (!fileInput.files.length) {
                showError('Please select at least one PDF resume.');
                return;
            }

            const formData = new FormData();
            formData.append('job_description', jd);
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files', fileInput.files[i]);
            }

            try {
                const response = await fetch(`${API_BASE}/unified/pipeline`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({ detail: response.statusText }));
                    throw new Error(`Server error: ${err.detail || response.status}`);
                }

                const data = await response.json();

                const resultContainer = document.getElementById('unified-result');
                const listContainer = document.getElementById('pipeline-list');
                listContainer.innerHTML = `<h3>Ranked Results (${data.count || 0} Candidates)</h3>`;

                if (data.results && data.results.length > 0) {
                    data.results.forEach(cand => {
                        const scoreColor = cand.total_score >= 70 ? '#10b981' : cand.total_score >= 50 ? '#f59e0b' : '#ef4444';
                        const item = document.createElement('div');
                        item.className = 'pipeline-item';
                        item.style.cssText = 'margin-bottom:1.5rem;border:1px solid var(--border);padding:1.25rem;border-radius:12px;background:rgba(255,255,255,0.02);';
                        item.innerHTML = `
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
                                <div>
                                    <span style="font-size:1.25rem;font-weight:700;color:var(--primary)">#${cand.rank}</span>
                                    <h3 style="display:inline;margin-left:10px;">${cand.name || 'Unknown'}</h3>
                                    <p style="font-size:0.9rem;color:var(--text-muted)">${cand.email || ''}</p>
                                </div>
                                <div style="text-align:right">
                                    <div style="font-size:1.5rem;font-weight:700;color:${scoreColor}">${cand.total_score}%</div>
                                    <div style="font-size:0.75rem;color:var(--text-muted)">Match Score</div>
                                </div>
                            </div>
                        `;
                        listContainer.appendChild(item);
                    });
                } else if (data.status === 'error') {
                    listContainer.innerHTML += `<p style="color:#ef4444">${data.message || 'No results returned.'}</p>`;
                }

                resultContainer.classList.remove('hidden');
            } catch (error) {
                showError('Pipeline Error: ' + error.message);
            } finally {
                hideLoader();  // ALWAYS hide loader after response
            }
        });
    }

    // --- CV Extract Form ---
    const extractForm = document.getElementById('extract-form');
    if (extractForm) {
        extractForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoader();

            const fileInput = document.getElementById('resume-file');
            if (!fileInput.files.length) {
                showError('Please select a PDF file.');
                return;
            }

            const includeAnalysis = document.getElementById('include-analysis').checked;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('include_analysis', includeAnalysis);

            try {
                const response = await fetch(`${API_BASE}/cv_extract/extract`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({ detail: response.statusText }));
                    throw new Error(`Server error: ${err.detail || response.status}`);
                }

                const data = await response.json();
                document.getElementById('extract-result').classList.remove('hidden');
                document.getElementById('extract-json').textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                showError('CV Extraction Error: ' + error.message);
            } finally {
                hideLoader();  // ALWAYS hide loader
            }
        });
    }

    // --- Resume Ranking Form ---
    const rankingForm = document.getElementById('ranking-form');
    if (rankingForm) {
        rankingForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            showLoader();

            const jobDescription = document.getElementById('job-description').value;
            const candidateDataRaw = document.getElementById('candidate-data').value;

            let candidateData;
            try {
                candidateData = JSON.parse(candidateDataRaw);
            } catch (e) {
                showError('Invalid JSON in Candidate Data field.');
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/resume_ranking/rank`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ job_description: jobDescription, candidates: candidateData })
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({ detail: response.statusText }));
                    throw new Error(`Server error: ${err.detail || response.status}`);
                }

                const data = await response.json();
                const resultContainer = document.getElementById('ranking-result');
                const listContainer = document.getElementById('ranking-list');
                listContainer.innerHTML = '';

                if (data.candidates && data.candidates.length > 0) {
                    data.candidates.forEach(cand => {
                        const item = document.createElement('div');
                        item.className = 'rank-item';
                        item.innerHTML = `
                            <div class="rank-number">#${cand.rank}</div>
                            <div class="rank-info">
                                <h3>${cand.name || 'Unknown'}</h3>
                                <p>${cand.email || ''} | ${cand.phone || ''}</p>
                                <small>${cand.skills_preview || ''}</small>
                            </div>
                            <div class="rank-score">${Math.round(cand.total_score)}%</div>
                        `;
                        listContainer.appendChild(item);
                    });
                } else {
                    listContainer.innerHTML = '<p>No candidates ranked.</p>';
                }
                resultContainer.classList.remove('hidden');
            } catch (error) {
                showError('Ranking Error: ' + error.message);
            } finally {
                hideLoader();  // ALWAYS hide loader
            }
        });
    }

    // Check server status on load
    const checkStatus = async () => {
        try {
            const resp = await fetch(`${API_BASE}/`);
            const data = await resp.json();
            const indicator = document.getElementById('status-indicator');
            if (data.status === 'running') {
                indicator.textContent = 'Server Online';
                indicator.style.color = 'var(--accent)';
            }
        } catch (e) {
            const indicator = document.getElementById('status-indicator');
            indicator.textContent = 'Server Offline';
            indicator.style.color = 'var(--error)';
            indicator.style.background = 'rgba(239,68,68,0.1)';
        }
    };

    checkStatus();
});
