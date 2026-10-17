import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Soccer Stats Docs',
  tagline: 'Computer vision for soccer statistics',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://capstone-computer-vision-for-soccer.vercel.app',
  baseUrl: '/',

  organizationName: 'mickeynotmouse',
  projectName: 'Capstone-Computer-Vision-for-Soccer-Statistics',

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
        },
        blog: {
          showReadingTime: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
  title: 'Home',
  logo: {
    alt: 'Home',
    src: 'img/logo.svg',
    href: '/docs/overview',
  },
  items: [
    {
      href: 'https://github.com/mickeynotmouse/Capstone-Computer-Vision-for-Soccer-Statistics',
      label: 'GitHub',
      position: 'right',
    },
  ],
},
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Overview', to: '/docs/overview'},
            {label: 'Architecture', to: '/docs/architecture'},
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/mickeynotmouse/Capstone-Computer-Vision-for-Soccer-Statistics',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Mikkel Jensen. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;