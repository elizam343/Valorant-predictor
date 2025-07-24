// Main JavaScript for Valorant Kill Line Predictor

document.addEventListener('DOMContentLoaded', function() {
    // Initialize searchable dropdowns
    initializeSearchableDropdowns();
    
    // Initialize prediction form
    initializePredictionForm();
    
    // Initialize loading states
    initializeLoadingStates();
});

function initializeSearchableDropdowns() {
    const dropdowns = document.querySelectorAll('.searchable-dropdown');
    
    dropdowns.forEach(dropdown => {
        const input = dropdown.querySelector('input');
        const list = dropdown.querySelector('.dropdown-list');
        const items = dropdown.querySelectorAll('.dropdown-item');
        
        if (!input || !list) return;
        
        // Show dropdown on focus
        input.addEventListener('focus', () => {
            list.classList.add('show');
        });
        
        // Hide dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!dropdown.contains(e.target)) {
                list.classList.remove('show');
            }
        });
        
        // Filter items on input
        input.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            let hasVisibleItems = false;
            
            items.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(searchTerm)) {
                    item.style.display = 'block';
                    hasVisibleItems = true;
                } else {
                    item.style.display = 'none';
                }
            });
            
            if (hasVisibleItems) {
                list.classList.add('show');
            } else {
                list.classList.remove('show');
            }
        });
        
        // Handle item selection
        items.forEach(item => {
            item.addEventListener('click', () => {
                input.value = item.textContent;
                input.dataset.value = item.dataset.value || item.textContent;
                list.classList.remove('show');
                
                // Trigger change event for form validation
                input.dispatchEvent(new Event('change'));
            });
            
            // Keyboard navigation
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    item.click();
                }
            });
        });
        
        // Keyboard navigation for input
        input.addEventListener('keydown', (e) => {
            const visibleItems = Array.from(items).filter(item => 
                item.style.display !== 'none'
            );
            const currentIndex = visibleItems.findIndex(item => 
                item.classList.contains('selected')
            );
            
            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    if (currentIndex < visibleItems.length - 1) {
                        if (currentIndex >= 0) {
                            visibleItems[currentIndex].classList.remove('selected');
                        }
                        visibleItems[currentIndex + 1].classList.add('selected');
                    }
                    break;
                    
                case 'ArrowUp':
                    e.preventDefault();
                    if (currentIndex > 0) {
                        visibleItems[currentIndex].classList.remove('selected');
                        visibleItems[currentIndex - 1].classList.add('selected');
                    }
                    break;
                    
                case 'Enter':
                    e.preventDefault();
                    if (currentIndex >= 0) {
                        visibleItems[currentIndex].click();
                    }
                    break;
                    
                case 'Escape':
                    list.classList.remove('show');
                    visibleItems.forEach(item => item.classList.remove('selected'));
                    break;
            }
        });
    });
}

function initializePredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    const playerInput = form.querySelector('input[name="player_name"]');
    const teamInput = form.querySelector('input[name="team"]');
    const opponentInput = form.querySelector('input[name="opponent_team"]');
    const mapInput = form.querySelector('select[name="map"]');
    const killLineInput = form.querySelector('input[name="kill_line"]');
    const submitBtn = form.querySelector('button[type="submit"]');
    const resultDiv = document.getElementById('prediction-result');

    // Enable player input when a team is selected
    if (teamInput && playerInput) {
        teamInput.addEventListener('change', () => {
            playerInput.disabled = false;
        });
    }

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!submitBtn || submitBtn.disabled) return;
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
        // Hide previous results
        if (resultDiv) {
            resultDiv.style.display = 'none';
        }
        try {
            // Use .dataset.value if present, else .value
            const data = {
                team: teamInput ? (teamInput.dataset.value || teamInput.value) : '',
                player_name: playerInput ? (playerInput.dataset.value || playerInput.value) : '',
                opponent_team: opponentInput ? (opponentInput.dataset.value || opponentInput.value) : '',
                map: mapInput ? mapInput.value : '',
                kill_line: killLineInput ? killLineInput.value : '',
                series_type: form.querySelector('[name="series_type"]').value,
                maps_scope: form.querySelector('[name="maps_scope"]').value,
                tournament: form.querySelector('[name="tournament"]').value
            };
            // Remove empty optional fields
            Object.keys(data).forEach(key => {
                if (data[key] === undefined || data[key] === null || data[key] === '') {
                    delete data[key];
                }
            });
            console.log('[Prediction Debug] Form data to send:', data);
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            console.log('[Prediction Debug] API response status:', response.status);
            const result = await response.json();
            console.log('[Prediction Debug] API response JSON:', result);
            if (result.success) {
                displayPredictionResult(result.prediction);
            } else {
                showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            console.error('[Prediction Debug] Prediction error:', error);
            showError('Network error. Please try again.');
        } finally {
            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = 'Get Prediction';
        }
    });
}

async function loadTeams(inputElement) {
    try {
        const response = await fetch('/api/teams');
        const result = await response.json();
        
        if (result.success) {
            const dropdown = inputElement.closest('.searchable-dropdown');
            const list = dropdown.querySelector('.dropdown-list');
            
            list.innerHTML = '';
            result.teams.forEach(team => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.textContent = team;
                item.dataset.value = team;
                list.appendChild(item);
            });
            
            // Reinitialize dropdown functionality
            initializeSearchableDropdowns();
        }
    } catch (error) {
        console.error('Error loading teams:', error);
    }
}

async function loadPlayersForTeam(team, inputElement) {
    console.log('[DEBUG] loadPlayersForTeam called with team:', team);
    console.log('[DEBUG] inputElement:', inputElement);
    
    try {
        const response = await fetch(`/api/players/${encodeURIComponent(team)}`);
        const result = await response.json();
        console.log('[DEBUG] API response:', result);
        
        if (result.success) {
            const dropdown = inputElement.closest('.searchable-dropdown');
            const list = dropdown.querySelector('.dropdown-list');
            console.log('[DEBUG] dropdown:', dropdown);
            console.log('[DEBUG] list:', list);
            
            // Clear and populate
            list.innerHTML = '';
            result.players.forEach(player => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.textContent = player.name;
                item.dataset.value = player.name;
                list.appendChild(item);
            });
            
            // Enable and clear input
            inputElement.disabled = false;
            inputElement.value = '';
            inputElement.dataset.value = '';
            
            // Force show dropdown
            list.classList.add('show');
            list.style.display = 'block';
            list.style.position = 'absolute';
            list.style.zIndex = '9999';
            list.style.background = 'white';
            list.style.border = '1px solid #ccc';
            list.style.width = '100%';
            
            inputElement.focus();
            
            console.log('[DEBUG] Loaded', result.players.length, 'players for team', team);
            console.log('[DEBUG] Dropdown HTML:', list.innerHTML);
            console.log('[DEBUG] Dropdown classList:', list.classList.toString());
            console.log('[DEBUG] Dropdown style.display:', list.style.display);
            console.log('[DEBUG] Input disabled:', inputElement.disabled);
        } else {
            console.log('[DEBUG] No players found for team:', team);
        }
    } catch (error) {
        console.error('[DEBUG] Error loading players:', error);
    }
}

function displayPredictionResult(prediction) {
    const resultDiv = document.getElementById('prediction-result');
    if (!resultDiv) return;

    // Check for required fields
    const requiredFields = [
        'player_name', 'team', 'kill_line', 'prediction', 'confidence',
        'recommended_action', 'over_probability', 'under_probability', 'unsure_probability'
    ];
    for (const field of requiredFields) {
        if (typeof prediction[field] === 'undefined') {
            showError(`Prediction result missing field: ${field}`);
            console.error('Prediction object missing field:', field, prediction);
            return;
        }
    }

    try {
        const confidencePercent = Math.round(prediction.confidence * 100);
        const predictionType = prediction.prediction.toLowerCase();

        resultDiv.innerHTML = `
            <div class="prediction-header">
                <h3>Prediction Result</h3>
                <span class="prediction-type ${predictionType}">${prediction.prediction}</span>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">${prediction.player_name}</div>
                    <div class="stat-label">Player</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${prediction.team}</div>
                    <div class="stat-label">Team</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${prediction.kill_line}</div>
                    <div class="stat-label">Kill Line</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${confidencePercent}%</div>
                    <div class="stat-label">Confidence</div>
                </div>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="mb-3">
                <strong>Recommended Action:</strong> ${prediction.recommended_action}
            </div>
            <div class="mb-3">
                <strong>Probabilities:</strong>
                <ul>
                    <li>Over: ${Math.round(prediction.over_probability * 100)}%</li>
                    <li>Under: ${Math.round(prediction.under_probability * 100)}%</li>
                    <li>Unsure: ${Math.round(prediction.unsure_probability * 100)}%</li>
                </ul>
            </div>
            ${prediction.explanation ? `
            <div class="mb-3">
                <strong>Analysis:</strong>
                <p>${prediction.explanation}</p>
            </div>
            ` : ''}
            ${prediction.player_stats ? `
            <div class="mb-3">
                <strong>Player Stats:</strong>
                <ul>
                    <li>Rating: ${prediction.player_stats.rating || 'N/A'}</li>
                    <li>Kills per Round: ${prediction.player_stats.kills_per_round || 'N/A'}</li>
                    <li>ACS: ${prediction.player_stats.average_combat_score || 'N/A'}</li>
                    <li>Headshot %: ${prediction.player_stats.headshot_percentage || 'N/A'}%</li>
                </ul>
            </div>
            ` : ''}
        `;
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        showError('Error displaying prediction result. See console for details.');
        console.error('Error rendering prediction result:', err, prediction);
    }
}

function showError(message) {
    const resultDiv = document.getElementById('prediction-result');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="alert alert-error">
                <strong>Error:</strong> ${message}
            </div>
        `;
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    }
    // Also log to console for debugging
    console.error('Prediction error:', message);
}

function initializeLoadingStates() {
    // Add loading states to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', function() {
            if (this.type === 'submit' && !this.disabled) {
                const originalText = this.textContent;
                this.disabled = true;
                this.innerHTML = '<span class="spinner"></span> Loading...';
                
                // Reset after a delay (for demo purposes)
                setTimeout(() => {
                    this.disabled = false;
                    this.textContent = originalText;
                }, 2000);
            }
        });
    });
}

// Utility functions
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatPercentage(num) {
    return `${Math.round(num * 100)}%`;
}

// Export functions for global access
window.ValorantPredictor = {
    displayPredictionResult,
    showError,
    loadTeams,
    loadPlayersForTeam
}; 