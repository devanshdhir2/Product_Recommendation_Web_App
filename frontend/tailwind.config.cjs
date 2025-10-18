/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                brand: {
                    50: "#f5f7fb",
                    100: "#e9eef7",
                    900: "#0b0e13"
                }
            }
        }
    },
    plugins: []
};
