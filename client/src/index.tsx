/* @refresh reload */
import './index.css';
import { render } from 'solid-js/web';
import { Route, RouteProps, Router } from '@solidjs/router';
import Home from './routes';
import { lazy } from 'solid-js';

const root = document.getElementById('root');

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
	throw new Error(
		'Root element not found. Did you forget to add it to your index.html? Or maybe the id attribute got misspelled?',
	);
}

// const routes: RouteProps<string, any> = {
// 	path: "/",
// 	component: lazy(() => import("./routes/index")),
// 	// children: [
// 	// 	{
// 	// 		path: "/auth"
// 	// 	}
// 	// ]
// }

render(
	() => (
		<Router>
			<Route path='/' component={Home} />
		</Router>
	)
	, root!
);
