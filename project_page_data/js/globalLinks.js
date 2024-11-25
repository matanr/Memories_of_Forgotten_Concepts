document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('a[href]').forEach(link => {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    });
});
