module.exports = {
    lang: 'en-US',
    title: 'Scikit-decide',
    description: 'This is scikit-decide documentation site',
    base: `/scikit-decide${process.env.DOCS_VERSION_PATH || '/'}`,

    locales: {
        '/': {
            lang: 'en-US',
            title: 'Scikit-decide',
            description: 'This is scikit-decide documentation',
        },
    },

    themeConfig: {
        repo: 'airbus/scikit-decide',
        logo: '/logo.svg',
        editLinks: false,
        docsDir: '',
        editLinkText: '',
        lastUpdated: false,
        locales: {
            '/': {
                selectLanguageName: 'en-US',
            },
        },
        nav: [
            {
                text: 'Home',
                link: '/'
            },
            {
                text: 'Guide',
                link: '/guide/'
            },
            {
                text: 'Reference',
                link: '/reference/'
            },
        ],
        sidebar: 'auto',
        markdown: {
            toc: {
                includeLevel: [2]
            }
        },
    },

    plugins: {
        '@vuepress/plugin-back-to-top': {},
        'mathjax': {
            target: 'svg',
            macros: { '*': '\\times' }
        },
    },
}
