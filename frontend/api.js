import { API_ENDPOINTS } from './config';

export const suggestPatch = async (issueId, diagnosis) => {
    try {
        const response = await fetch(API_ENDPOINTS.suggest_patch, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ issue_id: issueId, diagnosis }),
        });

        if (!response.ok) {
            throw new Error(`Failed with status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error suggesting patch:", error);
        throw error;
    }
};
