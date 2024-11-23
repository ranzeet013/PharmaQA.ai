async function askQuestion() {
    const query = document.getElementById('query').value;
    const responseElement = document.getElementById('response');
    
    const response = await fetch('http://127.0.0.1:5000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    });

    const data = await response.json();
    responseElement.textContent = "Answer: " + data.answer + "\nSources: " + data.sources.join(", ");
}
