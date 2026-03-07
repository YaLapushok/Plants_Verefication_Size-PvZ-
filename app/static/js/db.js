function openModal(src) {
    const modalImg = document.getElementById('modalImg');
    const imgModal = document.getElementById('imgModal');
    if (modalImg && imgModal) {
        modalImg.src = src;
        imgModal.classList.add('open');
    }
}

function closeModal(e) {
    const imgModal = document.getElementById('imgModal');
    if (imgModal && (e === undefined || e.target === imgModal)) {
        imgModal.classList.remove('open');
    }
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
});

// Checkbox Sync logic
document.addEventListener('DOMContentLoaded', () => {
    const galleryCheckboxes = document.querySelectorAll('.db-checkbox');
    const tableCheckboxes = document.querySelectorAll('.db-checkbox-table');

    galleryCheckboxes.forEach(cb => {
        cb.addEventListener('change', (e) => {
            const id = e.target.value;
            const tableCb = document.querySelector(`.db-checkbox-table[value="${id}"]`);
            if (tableCb) tableCb.checked = e.target.checked;
            updateSelectAllState();
        });
    });

    window.syncCheckboxes = function (tableCb) {
        const id = tableCb.value;
        const galleryCb = document.querySelector(`.db-checkbox[value="${id}"]`);
        if (galleryCb) galleryCb.checked = tableCb.checked;
        updateSelectAllState();
    };

    window.toggleAllCheckboxes = function (masterCb) {
        const isChecked = masterCb.checked;
        galleryCheckboxes.forEach(cb => cb.checked = isChecked);
        tableCheckboxes.forEach(cb => cb.checked = isChecked);
    };

    function updateSelectAllState() {
        const masterCb = document.getElementById('selectAllCheckbox');
        if (!masterCb) return;
        const allChecked = Array.from(tableCheckboxes).every(cb => cb.checked);
        const someChecked = Array.from(tableCheckboxes).some(cb => cb.checked);
        masterCb.checked = allChecked;
        masterCb.indeterminate = someChecked && !allChecked;
    }
});

// DB Export Logic
async function runExport(ids) {
    if (ids.length === 0) {
        alert("Нет данных для экспорта.");
        return;
    }

    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysis_ids: ids })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `plant_db_export_${new Date().toISOString().split('T')[0]}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            const error = await response.json();
            alert("Ошибка экспорта: " + (error.error || "Неизвестная ошибка"));
        }
    } catch (err) {
        console.error(err);
        alert("Сетевая ошибка при экспорте.");
    }
}

function exportSelected() {
    const selectedIds = Array.from(document.querySelectorAll('.db-checkbox:checked')).map(cb => parseInt(cb.value));
    if (selectedIds.length === 0) {
        alert("Выберите хотя бы один анализ для экспорта!");
        return;
    }
    runExport(selectedIds);
}

function exportAllDB() {
    const allIds = Array.from(document.querySelectorAll('.db-checkbox')).map(cb => parseInt(cb.value));
    runExport(allIds);
}
